require('dotenv').config();
require('tsx/cjs');

const cors = require('cors');
const express = require('express');
const { createClient } = require('@supabase/supabase-js');
const { generateMenuWithAgents } = require('./src/services/ai-agents.ts');

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
      return res.status(404).json({ error: 'Профиль не найден' });
    }

    const recipes = recipesResult.data || [];
    if (recipes.length === 0) {
      return res.status(409).json({ error: 'В таблице recipes нет блюд. Запустите npm run seed:recipes' });
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

    const preferredStore = profileResult.data.selected_stores?.[0] || 'Самокат';
    const compactProducts = (item) => ({
      store: item.store_name,
      search_term: item.search_term,
      title: item.product_title,
      price: item.price,
      pack_g: item.pack_weight_grams,
    });
    const inStock = (item) => item.in_stock !== false;
    let storeProducts = (productsResult.data || [])
      .filter((item) => item.store_name === preferredStore && inStock(item))
      .map(compactProducts);
    let activeStore = preferredStore;
    if (storeProducts.length === 0) {
      const samokat = (productsResult.data || [])
        .filter((item) => item.store_name === 'Самокат' && inStock(item))
        .map(compactProducts);
      if (samokat.length) {
        activeStore = 'Самокат';
        storeProducts = samokat;
      } else {
        storeProducts = (productsResult.data || []).filter(inStock).map(compactProducts);
        activeStore = storeProducts[0]?.store || preferredStore;
      }
    }

    const userPayload = {
      profile: {
        id: profileResult.data.id,
        budget_limit: Number(profileResult.data.budget_limit),
        selected_stores: profileResult.data.selected_stores || [],
        pricing_store: activeStore,
        diet_tags: profileResult.data.diet_tags || [],
        equipment_tags: profileResult.data.equipment_tags || [],
      },
      recipes: catalog,
      store_products: storeProducts,
    };

    let menu;
    let isFallback = false;
    try {
      menu = await generateMenuWithAgents(userPayload.profile, catalog, storeProducts);
      console.log(`[ai_success_rate] ok userId=${userId} store=${activeStore}`);
    } catch (agentsError) {
      isFallback = true;
      console.error(
        `[ai_success_rate] fail userId=${userId} store=${activeStore} error=`,
        agentsError instanceof Error ? agentsError.stack || agentsError.message : agentsError,
      );
      const fallback = buildFallbackMenu(catalog, userPayload.profile.budget_limit, activeStore);
      menu = normalizeMenu(fallback, catalog, userPayload.profile.budget_limit, activeStore);
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
    totalCost: Math.max(0, Math.round(Number(budgetLimit) * 0.85)),
    nutrition: null,
    zeroWasteNotes: 'Резервное меню: ответ ИИ был повреждён, собрали рацион из каталога рецептов.',
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
    totalCost,
    nutrition: aiJson.nutrition || null,
    zeroWasteNotes: aiJson.zeroWasteNotes || aiJson.zero_waste_notes || '',
    days,
  };
}

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Chef API listening on http://0.0.0.0:${PORT} (PORT=${process.env.PORT ?? 'default'})`);
});
