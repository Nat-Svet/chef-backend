require('dotenv').config();

const cors = require('cors');
const express = require('express');
const { createClient } = require('@supabase/supabase-js');

const PORT = Number(process.env.PORT) || 8080;
const TIMEWEB_AI_URL = (process.env.TIMEWEB_AI_URL || 'https://api.timeweb.ai/v1').replace(/\/$/, '');
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

if (!SUPABASE_URL || !SUPABASE_KEY) {
  throw new Error('Задайте SUPABASE_URL / EXPO_PUBLIC_SUPABASE_URL и ключ anon в .env');
}

if (!TIMEWEB_AI_KEY) {
  throw new Error('Задайте TIMEWEB_AI_KEY в .env');
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY, {
  auth: { persistSession: false, autoRefreshToken: false },
});

const WEEK_DAYS = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'];

const SYSTEM_PROMPT = `Ты — оркестратор трёх ИИ-агентов приложения «Шеф в Кармане». Работай строго по ролям и верни ОДИН JSON-объект без markdown и без комментариев.

РОЛЬ 1. Агент-Диетолог
- Фильтруй рецепты по diet_tags профиля (теги рецепта должны пересекаться с целями, если цели заданы).
- Считай суммарное КБЖУ недели по выбранным блюдам (ккал, белки, жиры, углеводы на 1 порцию × 21 приём пищи).

РОЛЬ 2. Агент-Шеф
- Оставляй только блюда, чьи tags совместимы с equipment_tags профиля (нужная техника есть у пользователя).
- Составь меню на 7 дней: завтрак, обед, ужин (ровно 21 слот).
- Предпочитай неповторяющиеся recipe id. Если в каталоге меньше 21 уникального рецепта, чередуй доступные id, НЕ выдумывай новые id.
- meal_type должен соответствовать слоту: завтрак / обед / ужин. Если подходящего типа нет — возьми ближайший допустимый id из каталога.

РОЛЬ 3. Агент-Закупщик
- Используй ТОЛЬКО товары store_products магазина pricing_store (если выбранный магазин без цен, бери переданный каталог).
- Сопоставляй ингредиенты по search_term.
- Считай корзину на 7 дней (1 порция на приём пищи), принцип Zero Waste: остаток упаковки идёт в другие блюда недели, не покупай лишние упаковки без нужды.
- Итоговая стоимость (totalCost) СТРОГО <= budget_limit. Если не укладываешься — замени дорогие блюда на более дешёвые из каталога.

ФОРМАТ ОТВЕТА (только JSON):
{
  "store": "Самокат",
  "totalCost": 0,
  "nutrition": { "kcal": 0, "protein": 0, "fat": 0, "carb": 0 },
  "zeroWasteNotes": "кратко",
  "days": [
    {
      "day": "Пн",
      "breakfastId": 1,
      "lunchId": 2,
      "dinnerId": 3
    }
  ]
}

Правила JSON:
- days: ровно 7 объектов, day строго: Пн, Вт, Ср, Чт, Пт, Сб, Вс.
- breakfastId, lunchId, dinnerId — целые id из переданного каталога recipes.
- totalCost — число в рублях, не больше budget_limit.`;

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
  res.json({ ok: true, service: 'chef-v-karmane-api' });
});

app.post('/api/generate-menu', async (req, res) => {
  const userId = req.body?.userId;

  if (!userId || typeof userId !== 'string') {
    return res.status(400).json({ error: 'Передайте { userId } — UUID профиля из таблицы profiles' });
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

    const aiRaw = await requestTimewebMenu(userPayload);
    const menu = normalizeMenu(aiRaw, catalog, userPayload.profile.budget_limit, activeStore);

    return res.json({
      userId,
      model: TIMEWEB_AI_MODEL,
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

async function requestTimewebMenu(payload) {
  const messages = [
    { role: 'system', content: SYSTEM_PROMPT },
    {
      role: 'user',
      content: `Собери меню на неделю по данным:\n${JSON.stringify(payload)}\n\nВерни сразу финальный JSON, без рассуждений и без markdown.`,
    },
  ];

  const baseBody = {
    model: TIMEWEB_AI_MODEL,
    temperature: 0.1,
    max_tokens: 8192,
    messages,
  };

  let body = await callTimewebChat({ ...baseBody, response_format: { type: 'json_object' } });
  if (body.error) {
    body = await callTimewebChat(baseBody);
  }

  if (body.error) {
    const details = body.error?.message || body.message || JSON.stringify(body);
    throw new Error(`Timeweb AI: ${details}`);
  }

  try {
    return extractMenuJson(body);
  } catch (firstError) {
    const repair = await callTimewebChat({
      model: TIMEWEB_AI_MODEL,
      temperature: 0,
      max_tokens: 2048,
      messages: [
        ...messages,
        { role: 'assistant', content: extractRawContent(body) || '' },
        {
          role: 'user',
          content:
            'Ответ невалиден. Верни ТОЛЬКО JSON-объект с полями store, totalCost, nutrition, zeroWasteNotes, days. Без текста вокруг.',
        },
      ],
    });

    if (repair.error) {
      throw firstError;
    }

    return extractMenuJson(repair);
  }
}

function extractRawContent(body) {
  const message = body.choices?.[0]?.message || {};
  const pick = (value) => {
    if (!value) return '';
    if (typeof value === 'string') return value;
    if (Array.isArray(value)) {
      return value.map((part) => part.text || part.content || '').join('\n');
    }
    return value.text || value.content || '';
  };

  return (pick(message.content) || pick(message.reasoning_content)).trim();
}

function extractMenuJson(body) {
  const content = extractRawContent(body);
  if (!content) {
    throw new Error('Пустой ответ модели');
  }
  return parseJsonContent(content);
}

async function callTimewebChat(payload) {
  const response = await fetch(`${TIMEWEB_AI_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${TIMEWEB_AI_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    return { error: body.error || { message: `HTTP ${response.status}` }, status: response.status };
  }
  return body;
}

function parseJsonContent(content) {
  const trimmed = String(content).trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const raw = (fenced ? fenced[1] : trimmed).trim();

  try {
    return JSON.parse(raw);
  } catch {
    const start = raw.indexOf('{');
    const end = raw.lastIndexOf('}');
    if (start >= 0 && end > start) {
      return JSON.parse(raw.slice(start, end + 1));
    }
    throw new Error('Модель вернула не JSON');
  }
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
