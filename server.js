require('dotenv').config();
require('tsx/cjs');

const cors = require('cors');
const express = require('express');
const { createClient } = require('@supabase/supabase-js');
const { generateMenuWithAgents, pickReplacementRecipe } = require('./src/services/ai-agents.ts');

const PORT = Number(process.env.PORT) || 8080;
const TIMEWEB_AI_KEY = process.env.TIMEWEB_AI_KEY;
const MODEL_ALIASES = {
  'deepseek-v4-flash': 'deepseek/deepseek-v4-flash',
  'deepseek-v4-pro': 'deepseek/deepseek-v4-pro',
};
const TIMEWEB_AI_MODEL =
  MODEL_ALIASES[process.env.TIMEWEB_AI_MODEL] ||
  process.env.TIMEWEB_AI_MODEL ||
  'deepseek/deepseek-v4-flash';
const SUPABASE_URL = process.env.SUPABASE_URL || process.env.EXPO_PUBLIC_SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_ANON_KEY || process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;

const supabase =
  SUPABASE_URL && SUPABASE_KEY
    ? createClient(SUPABASE_URL, SUPABASE_KEY, {
        auth: { persistSession: false, autoRefreshToken: false },
      })
    : null;

const WEEK_DAYS = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'];

const app = express();

const corsOptions = {
  origin: true,
  methods: ['GET', 'HEAD', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'Accept'],
  optionsSuccessStatus: 204,
};

app.use(cors(corsOptions));
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', req.headers.origin || '*');
  res.header('Access-Control-Allow-Methods', 'GET,HEAD,POST,OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(204);
  }
  next();
});
app.use(express.json({ limit: '1mb' }));

app.get(['/', '/health'], (_req, res) => {
  res.status(200).json({
    ok: true,
    service: 'chef-v-karmane-api',
    port: PORT,
    hasSupabase: Boolean(supabase),
    hasAiKey: Boolean(TIMEWEB_AI_KEY),
  });
});

/** Общая загрузка профиля + каталога рецептов + товаров выбранных магазинов.
 *  Используется и генерацией меню, и точечной заменой одного блюда. */
async function loadProfileAndCatalog(userId) {
  const [profileResult, recipesResult, ingredientsResult, productsResult] = await Promise.all([
    supabase.from('profiles').select('*').eq('id', userId).maybeSingle(),
    supabase.from('recipes').select('id, title, cooking_time, meal_type, tags'),
    supabase.from('recipe_ingredients').select('recipe_id, name, amount_grams, kcal, protein, fat, carb'),
    supabase.from('store_products').select('store_name, search_term, product_title, price, sku_id, pack_weight_grams, in_stock'),
  ]);

  if (profileResult.error) throw profileResult.error;
  if (recipesResult.error) throw recipesResult.error;
  if (ingredientsResult.error) throw ingredientsResult.error;
  if (productsResult.error) throw productsResult.error;

  if (!profileResult.data) {
    return { error: { status: 404, message: 'Профиль не найден' } };
  }

  const recipes = recipesResult.data || [];
  if (recipes.length === 0) {
    return { error: { status: 409, message: 'В таблице recipes нет блюд. Запустите npm run seed:recipes' } };
  }

  const catalog = recipes.map((recipe) => ({
    id: recipe.id,
    title: recipe.title,
    cooking_time: recipe.cooking_time,
    meal_type: recipe.meal_type,
    tags: recipe.tags,
    ingredients: (ingredientsResult.data || [])
      .filter((item) => item.recipe_id === recipe.id)
      .map((item) => ({
        name: item.name,
        grams: item.amount_grams,
        kcal: item.kcal,
        protein: item.protein,
        fat: item.fat,
        carb: item.carb,
      })),
  }));

  // MVP: один магазин. Берём первый (единственный) выбранный; если у него нет
  // товаров в каталоге — запасной вариант Самокат, чтобы корзина не осталась пустой.
  const selectedStore = profileResult.data.selected_stores?.[0] || 'Самокат';
  const compactProducts = (item) => ({
    store: item.store_name,
    search_term: item.search_term,
    title: item.product_title,
    price: item.price,
    pack_g: item.pack_weight_grams,
  });
  const inStock = (item) => item.in_stock !== false;
  const productsOf = (store) =>
    (productsResult.data || []).filter((item) => item.store_name === store && inStock(item)).map(compactProducts);

  let activeStore = selectedStore;
  let storeProducts = productsOf(selectedStore);
  if (storeProducts.length === 0 && selectedStore !== 'Самокат') {
    activeStore = 'Самокат';
    storeProducts = productsOf('Самокат');
  }

  const profile = {
    id: profileResult.data.id,
    budget_limit: Number(profileResult.data.budget_limit),
    selected_stores: [activeStore],
    pricing_store: activeStore,
    // MVP: ровно одна главная цель рациона (старые профили с несколькими — берём первую)
    diet_tags: (profileResult.data.diet_tags || []).slice(0, 1),
    portions: 1,
    equipment_tags: profileResult.data.equipment_tags || [],
  };

  return { profile, catalog, storeProducts };
}

app.post('/api/generate-menu', async (req, res) => {
  const userId = req.body?.userId;

  if (!userId || typeof userId !== 'string') {
    return res.status(400).json({ error: 'Передайте { userId } — UUID профиля из таблицы profiles' });
  }

  if (!supabase) {
    return res.status(500).json({
      error: 'Не заданы SUPABASE_URL и SUPABASE_ANON_KEY в переменных Timeweb',
    });
  }

  if (!TIMEWEB_AI_KEY) {
    return res.status(500).json({
      error: 'Не задан TIMEWEB_AI_KEY в переменных Timeweb',
    });
  }

  try {
    const loaded = await loadProfileAndCatalog(userId);
    if (loaded.error) {
      return res.status(loaded.error.status).json({ error: loaded.error.message });
    }
    const { profile, catalog, storeProducts } = loaded;
    profile.portions = Math.min(20, Math.max(1, Math.round(Number(req.body?.portions) || 1)));

    let menu;
    let isFallback = false;
    try {
      menu = await generateMenuWithAgents(profile, catalog, storeProducts);
      console.log(`[ai_success_rate] ok userId=${userId} store=${profile.pricing_store}`);
    } catch (agentsError) {
      isFallback = true;
      console.error(
        `[ai_success_rate] fail userId=${userId} store=${profile.pricing_store} error=`,
        agentsError instanceof Error ? agentsError.stack || agentsError.message : agentsError,
      );
      const fallback = buildFallbackMenu(catalog, profile.budget_limit, profile.pricing_store);
      menu = normalizeMenu(fallback, catalog, profile.budget_limit, profile.pricing_store);
    }

    return res.json({
      userId,
      model: TIMEWEB_AI_MODEL,
      isFallback,
      menu,
    });
  } catch (error) {
    console.error('generate-menu failed:', error);
    return res.status(502).json({
      error: 'Не удалось сгенерировать меню',
      details: error instanceof Error ? error.message : String(error),
    });
  }
});

app.post('/api/regenerate-meal', async (req, res) => {
  const { userId, mealType, excludeIds, rejectedIds, siblingIds } = req.body || {};

  if (!userId || typeof userId !== 'string') {
    return res.status(400).json({ error: 'Передайте { userId }' });
  }
  if (!['завтрак', 'обед', 'ужин'].includes(mealType)) {
    return res.status(400).json({ error: 'mealType должен быть завтрак/обед/ужин' });
  }
  if (!supabase) {
    return res.status(500).json({ error: 'Не заданы SUPABASE_URL и SUPABASE_ANON_KEY в переменных Timeweb' });
  }

  try {
    const loaded = await loadProfileAndCatalog(userId);
    if (loaded.error) {
      return res.status(loaded.error.status).json({ error: loaded.error.message });
    }

    const exclude = Array.isArray(excludeIds) ? excludeIds.map(Number).filter(Number.isFinite) : [];
    const toIds = (list) => (Array.isArray(list) ? list.map(Number).filter(Number.isFinite) : []);
    const recipeId = pickReplacementRecipe(
      loaded.profile,
      loaded.catalog,
      mealType,
      exclude,
      toIds(rejectedIds),
      toIds(siblingIds),
    );

    if (recipeId === null) {
      return res.status(409).json({ error: 'Нет подходящих блюд под текущие фильтры' });
    }

    return res.json({ recipeId });
  } catch (error) {
    console.error('regenerate-meal failed:', error);
    return res.status(502).json({
      error: 'Не удалось подобрать замену',
      details: error instanceof Error ? error.message : String(error),
    });
  }
});

function buildFallbackMenu(catalog, budgetLimit, store) {
  const byMeal = {
    завтрак: catalog.filter((item) => item.meal_type === 'завтрак'),
    обед: catalog.filter((item) => item.meal_type === 'обед'),
    ужин: catalog.filter((item) => item.meal_type === 'ужин'),
  };

  const pickId = (mealType, index) => {
    const pool = byMeal[mealType].length ? byMeal[mealType] : catalog;
    return Number(pool[index % pool.length].id);
  };

  return {
    store,
    stores: [store],
    totalCost: Math.max(0, Math.round(Number(budgetLimit) * 0.85)),
    nutrition: null,
    zeroWasteNotes: 'Собрали сбалансированный рацион из нашего проверенного каталога рецептов.',
    scarcityNotice: null,
    days: WEEK_DAYS.map((day, index) => ({
      day,
      breakfastId: pickId('завтрак', index),
      lunchId: pickId('обед', index),
      dinnerId: pickId('ужин', index),
    })),
  };
}

function normalizeMenu(aiJson, catalog, budgetLimit, fallbackStore) {
  const ids = new Set(catalog.map((item) => Number(item.id)));
  const byMeal = {
    завтрак: catalog.filter((item) => item.meal_type === 'завтрак').map((item) => Number(item.id)),
    обед: catalog.filter((item) => item.meal_type === 'обед').map((item) => Number(item.id)),
    ужин: catalog.filter((item) => item.meal_type === 'ужин').map((item) => Number(item.id)),
  };

  const pick = (value, mealType, index) => {
    const numeric = Number(value);
    if (ids.has(numeric)) return numeric;
    const pool = byMeal[mealType].length ? byMeal[mealType] : [...ids];
    return pool[index % pool.length];
  };

  const sourceDays = Array.isArray(aiJson.days) ? aiJson.days : [];
  let days = WEEK_DAYS.map((day, index) => {
    const row = sourceDays.find((item) => item.day === day) || sourceDays[index] || {};
    return {
      day,
      breakfastId: pick(row.breakfastId ?? row.breakfast_id, 'завтрак', index),
      lunchId: pick(row.lunchId ?? row.lunch_id, 'обед', index),
      dinnerId: pick(row.dinnerId ?? row.dinner_id, 'ужин', index),
    };
  });

  const uniqueMeals = new Set(days.flatMap((item) => [item.breakfastId, item.lunchId, item.dinnerId]));
  if (uniqueMeals.size === 1 && catalog.length > 1) {
    days = WEEK_DAYS.map((day, index) => ({
      day,
      breakfastId: pick(null, 'завтрак', index),
      lunchId: pick(null, 'обед', index),
      dinnerId: pick(null, 'ужин', index),
    }));
  }

  let totalCost = Number(aiJson.totalCost ?? aiJson.total_cost ?? 0);
  if (!Number.isFinite(totalCost) || totalCost < 0) totalCost = 0;
  if (totalCost > budgetLimit) totalCost = budgetLimit;

  return {
    store: aiJson.store || fallbackStore,
    stores: Array.isArray(aiJson.stores) && aiJson.stores.length ? aiJson.stores : [aiJson.store || fallbackStore],
    totalCost,
    nutrition: aiJson.nutrition || null,
    zeroWasteNotes: aiJson.zeroWasteNotes || aiJson.zero_waste_notes || '',
    scarcityNotice: aiJson.scarcityNotice ?? null,
    days,
  };
}

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Chef API listening on http://0.0.0.0:${PORT} (PORT=${process.env.PORT ?? 'default'})`);
});
