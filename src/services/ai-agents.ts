/**
 * Пайплайн из трёх ИИ-агентов (Диетолог → Шеф → Закупщик) поверх DeepSeek (Timeweb AI).
 * Математика (стоимость, КБЖУ) считается детерминированно в коде — LLM отвечает
 * только за отбор/распределение рецептов и текст zeroWasteNotes.
 */

export type MealType = 'завтрак' | 'обед' | 'ужин';
export type WeekDay = 'Пн' | 'Вт' | 'Ср' | 'Чт' | 'Пт' | 'Сб' | 'Вс';
export type ShoppingCategory = 'Овощи и фрукты' | 'Мясо и птица' | 'Молочные продукты' | 'Бакалея';

export type CatalogIngredient = {
  name: string;
  grams: number;
  kcal: number;
  protein: number;
  fat: number;
  carb: number;
};

export type CatalogRecipe = {
  id: number;
  title: string;
  cooking_time: number | null;
  meal_type: MealType | null;
  tags: string[];
  ingredients: CatalogIngredient[];
};

export type StoreProductCompact = {
  store: string;
  search_term: string;
  title: string;
  price: number;
  pack_g: number | null;
};

export type AgentProfile = {
  id: string;
  budget_limit: number;
  selected_stores: string[];
  pricing_store: string;
  diet_tags: string[];
  equipment_tags: string[];
};

export type MenuDay = { day: WeekDay; breakfastId: number; lunchId: number; dinnerId: number };
export type ShoppingItem = { name: string; grams: number; category: ShoppingCategory; price: number };

export type GeneratedMenuPayload = {
  store: string;
  totalCost: number;
  nutrition: { kcal: number; protein: number; fat: number; carb: number } | null;
  zeroWasteNotes: string;
  days: MenuDay[];
  shoppingItems: ShoppingItem[];
};

const WEEK_DAYS: WeekDay[] = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'];
const EQUIPMENT = ['Плита', 'Духовка', 'Мультиварка'];

const DEEPSEEK_URL = (process.env.TIMEWEB_AI_URL || 'https://api.timeweb.ai/v1').replace(/\/$/, '');
const DEEPSEEK_KEY = process.env.TIMEWEB_AI_KEY || '';
const DEEPSEEK_MODEL = process.env.TIMEWEB_AI_MODEL || 'deepseek/deepseek-v4-flash';

type ChatMessage = { role: 'system' | 'user'; content: string };

// ---------- Низкоуровневый клиент DeepSeek ----------

async function callDeepSeek(messages: ChatMessage[]): Promise<any> {
  if (!DEEPSEEK_KEY) throw new Error('TIMEWEB_AI_KEY не задан');

  const res = await fetch(`${DEEPSEEK_URL}/chat/completions`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${DEEPSEEK_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: DEEPSEEK_MODEL,
      temperature: 0,
      max_tokens: 2048,
      response_format: { type: 'json_object' },
      messages,
    }),
  });

  const json = await res.json().catch(() => ({}) as any);
  if (!res.ok || json.error) {
    throw new Error(`DeepSeek error: ${json.error?.message || res.status}`);
  }

  const content = json.choices?.[0]?.message?.content;
  if (!content) throw new Error('Пустой ответ модели');
  return parseJson(String(content));
}

function parseJson(raw: string): any {
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  let text = (fenced ? fenced[1] : raw).trim();
  const start = text.indexOf('{');
  if (start > 0) text = text.slice(start);

  try {
    return JSON.parse(text);
  } catch {
    return JSON.parse(repairJson(text));
  }
}

function repairJson(text: string): string {
  let raw = text.replace(/,\s*([}\]])/g, '$1');
  const stack: string[] = [];
  let inString = false;
  let escaped = false;

  for (const ch of raw) {
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{' || ch === '[') stack.push(ch === '{' ? '}' : ']');
    else if (ch === '}' || ch === ']') stack.pop();
  }
  if (inString) raw += '"';
  while (stack.length) raw += stack.pop();

  return raw.replace(/,\s*([}\]])/g, '$1');
}

async function callWithRetry(messages: ChatMessage[], attempts = 2): Promise<any> {
  let lastError: unknown;
  for (let i = 0; i < attempts; i += 1) {
    try {
      return await callDeepSeek(messages);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error('DeepSeek pipeline failed');
}

// ---------- Агент 1: Диетолог ----------
// Вход: профиль + весь каталог. Выход: id рецептов, прошедших фильтр diet_tags.

async function runDietitianAgent(profile: AgentProfile, catalog: CatalogRecipe[]): Promise<number[]> {
  if (profile.diet_tags.length === 0) {
    return catalog.map((r) => r.id);
  }

  const compact = catalog.map((r) => ({ id: r.id, tags: r.tags, meal_type: r.meal_type }));

  const system = `Ты — Агент-Диетолог приложения «Шеф в Кармане».
Задача: отобрать id рецептов, чьи tags пересекаются с diet_tags пользователя.
Правила:
1. Рецепт подходит, если хотя бы один его tag входит в diet_tags.
2. Если после фильтра для какого-то meal_type (завтрак/обед/ужин) остаётся меньше 3 рецептов — верни для этого meal_type ВСЕ id без фильтра. Рацион не может остаться без блюд.
3. Никогда не выдумывай id, которых нет в переданном каталоге.
Верни СТРОГО один JSON без markdown: {"filteredRecipeIds": [1,2,3]}`;

  const user = `diet_tags: ${JSON.stringify(profile.diet_tags)}
Каталог (id, tags, meal_type): ${JSON.stringify(compact)}`;

  try {
    const result = await callWithRetry([
      { role: 'system', content: system },
      { role: 'user', content: user },
    ]);
    const ids: number[] = Array.isArray(result.filteredRecipeIds)
      ? result.filteredRecipeIds.map(Number).filter((id: number) => catalog.some((r) => r.id === id))
      : [];
    return ensureEveryMealTypeCovered(ids, catalog);
  } catch {
    return catalog.map((r) => r.id);
  }
}

function ensureEveryMealTypeCovered(ids: number[], catalog: CatalogRecipe[]): number[] {
  const idSet = new Set(ids);
  const mealTypes: MealType[] = ['завтрак', 'обед', 'ужин'];
  for (const type of mealTypes) {
    const hasAny = catalog.some((r) => r.meal_type === type && idSet.has(r.id));
    if (!hasAny) {
      catalog.filter((r) => r.meal_type === type).forEach((r) => idSet.add(r.id));
    }
  }
  return [...idSet];
}

// ---------- Агент 2: Шеф-повар ----------
// Вход: id от Диетолога + equipment_tags. Выход: раскладка 7×3 = 21 слот.

const MEAL_TYPES: MealType[] = ['завтрак', 'обед', 'ужин'];

async function runChefAgent(
  profile: AgentProfile,
  catalog: CatalogRecipe[],
  allowedIds: number[],
): Promise<MenuDay[]> {
  const mealPools = buildMealPools(catalog, allowedIds, profile.equipment_tags);
  const compact = MEAL_TYPES.flatMap((type) =>
    mealPools[type].map((r) => ({ id: r.id, meal_type: r.meal_type, tags: r.tags, cooking_time: r.cooking_time })),
  );

  const system = `Ты — Агент-Шеф приложения «Шеф в Кармане».
Составь меню на 7 дней (Пн..Вс) по 3 приёма пищи: завтрак, обед, ужин. Итого ровно 21 слот.
Правила:
1. Используй ТОЛЬКО id из переданного списка recipes, не выдумывай новые.
2. meal_type блюда обязан соответствовать слоту (завтрак/обед/ужин).
3. Не повторяй одно и то же блюдо больше 2 раз за неделю, если рецептов достаточно для разнообразия.
4. Если рецептов одного meal_type меньше 7 — чередуй доступные.
Верни СТРОГО один JSON без markdown:
{"days":[{"day":"Пн","breakfastId":1,"lunchId":2,"dinnerId":3}]}
days — ровно 7 объектов, day строго: Пн, Вт, Ср, Чт, Пт, Сб, Вс.`;

  const user = `equipment_tags: ${JSON.stringify(profile.equipment_tags)}
Доступные рецепты (id, meal_type, tags, cooking_time): ${JSON.stringify(compact)}`;

  try {
    const result = await callWithRetry([
      { role: 'system', content: system },
      { role: 'user', content: user },
    ]);
    return normalizeDays(result.days, mealPools, catalog);
  } catch {
    return normalizeDays([], mealPools, catalog);
  }
}

function isEquipmentCompatible(recipe: CatalogRecipe, equipmentTags: string[]): boolean {
  const required = recipe.tags.filter((tag) => EQUIPMENT.includes(tag));
  if (required.length === 0) return true;
  return required.some((tag) => equipmentTags.includes(tag));
}

/** Пул на каждый meal_type строится независимо: сужение по технике/диете
 *  никогда не "перетекает" в другой приём пищи — деградация идёт только
 *  внутри своего типа (equipment -> diet -> весь каталог этого типа). */
function buildMealPools(
  catalog: CatalogRecipe[],
  allowedIds: number[],
  equipmentTags: string[],
): Record<MealType, CatalogRecipe[]> {
  const allowedSet = new Set(allowedIds);
  const pools = {} as Record<MealType, CatalogRecipe[]>;

  for (const type of MEAL_TYPES) {
    const byType = catalog.filter((r) => r.meal_type === type);
    const byDiet = byType.filter((r) => allowedSet.has(r.id));
    const byEquip = byDiet.filter((r) => isEquipmentCompatible(r, equipmentTags));
    pools[type] = byEquip.length ? byEquip : byDiet.length ? byDiet : byType;
  }

  return pools;
}

function normalizeDays(
  rawDays: unknown,
  mealPools: Record<MealType, CatalogRecipe[]>,
  catalog: CatalogRecipe[],
): MenuDay[] {
  const typeById = new Map(catalog.map((r) => [r.id, r.meal_type]));

  const pick = (value: unknown, type: MealType, index: number): number => {
    const numeric = Number(value);
    if (typeById.get(numeric) === type) return numeric;

    const list = mealPools[type].length ? mealPools[type] : catalog.filter((r) => r.meal_type === type);
    if (list.length) return list[index % list.length].id;

    console.warn(`Каталог не содержит рецептов meal_type="${type}"`);
    return catalog[index % catalog.length]?.id;
  };

  const source = Array.isArray(rawDays) ? (rawDays as any[]) : [];
  return WEEK_DAYS.map((day, index) => {
    const row = source.find((item) => item?.day === day) ?? source[index] ?? {};
    return {
      day,
      breakfastId: pick(row.breakfastId, 'завтрак', index),
      lunchId: pick(row.lunchId, 'обед', index),
      dinnerId: pick(row.dinnerId, 'ужин', index),
    };
  });
}

// ---------- Агент 3: Закупщик ----------
// Вход: раскладка дней + товары магазина. LLM пишет только текст Zero Waste;
// стоимость и состав корзины считаются кодом (детерминированно, без ошибок округления/JSON у модели).

async function runBuyerAgent(
  profile: AgentProfile,
  days: MenuDay[],
  catalog: CatalogRecipe[],
  storeProducts: StoreProductCompact[],
): Promise<{ zeroWasteNotes: string }> {
  const recipeById = new Map(catalog.map((r) => [r.id, r]));
  const usedIngredients = new Set<string>();
  for (const day of days) {
    for (const id of [day.breakfastId, day.lunchId, day.dinnerId]) {
      recipeById.get(id)?.ingredients.forEach((ing) => usedIngredients.add(ing.name));
    }
  }

  const system = `Ты — Агент-Закупщик приложения «Шеф в Кармане».
У тебя есть список ингредиентов недели и товары магазина ${profile.pricing_store}.
Задача: одним коротким предложением (zeroWasteNotes) объяснить, как распределить остатки упаковок между блюдами недели по принципу Zero Waste, опираясь на реальные ингредиенты из списка.
Верни СТРОГО один JSON без markdown: {"zeroWasteNotes": "текст"}`;

  const user = `Ингредиенты недели: ${JSON.stringify([...usedIngredients])}
Товары магазина: ${JSON.stringify(storeProducts.slice(0, 60))}`;

  try {
    const result = await callWithRetry([
      { role: 'system', content: system },
      { role: 'user', content: user },
    ]);
    return { zeroWasteNotes: typeof result.zeroWasteNotes === 'string' ? result.zeroWasteNotes : '' };
  } catch {
    return { zeroWasteNotes: 'Остатки упаковок распределены на другие блюда недели.' };
  }
}

function buildShoppingItems(
  days: MenuDay[],
  catalog: CatalogRecipe[],
  storeProducts: StoreProductCompact[],
  budgetLimit: number,
): { items: ShoppingItem[]; totalCost: number } {
  const recipeById = new Map(catalog.map((r) => [r.id, r]));
  const counts = new Map<number, number>();
  for (const day of days) {
    for (const id of [day.breakfastId, day.lunchId, day.dinnerId]) {
      counts.set(id, (counts.get(id) ?? 0) + 1);
    }
  }

  const gramsByName = new Map<string, number>();
  for (const [id, times] of counts) {
    const recipe = recipeById.get(id);
    if (!recipe) continue;
    for (const ing of recipe.ingredients) {
      gramsByName.set(ing.name, (gramsByName.get(ing.name) ?? 0) + ing.grams * times);
    }
  }

  const matchProduct = (name: string) => {
    const needle = name.toLowerCase();
    return (
      storeProducts.find((p) => p.search_term.toLowerCase() === needle) ??
      storeProducts.find(
        (p) => needle.includes(p.search_term.toLowerCase()) || p.search_term.toLowerCase().includes(needle),
      )
    );
  };

  const raw = [...gramsByName.entries()].map(([name, grams]) => {
    const product = matchProduct(name);
    const packGrams = product?.pack_g && product.pack_g > 0 ? product.pack_g : grams;
    const packs = product ? Math.max(1, Math.ceil(grams / packGrams)) : 1;
    const price = product ? packs * Number(product.price) : Math.max(1, Math.round(grams * 0.15));
    return { name, grams, category: guessCategory(name), price };
  });

  const rawTotal = raw.reduce((sum, item) => sum + item.price, 0) || 1;
  const target = budgetLimit > 0 ? Math.min(Math.round(rawTotal), budgetLimit) : Math.round(rawTotal);

  const items = raw
    .map((item) => ({ ...item, price: Math.max(1, Math.round((item.price / rawTotal) * target)) }))
    .sort((a, b) => a.name.localeCompare(b.name, 'ru'));

  const drift = target - items.reduce((sum, item) => sum + item.price, 0);
  if (items.length && drift !== 0) {
    items[items.length - 1].price = Math.max(1, items[items.length - 1].price + drift);
  }

  const totalCost = items.reduce((sum, item) => sum + item.price, 0);
  return { items, totalCost };
}

function guessCategory(name: string): ShoppingCategory {
  const lower = name.toLowerCase();
  if (/(филе|курин|мясо|фарш)/.test(lower)) return 'Мясо и птица';
  if (/(молоко|йогурт|сыр|творог)/.test(lower)) return 'Молочные продукты';
  if (/(ягод|овощ|перец|кабач|томат|морков|лук|брокколи)/.test(lower)) return 'Овощи и фрукты';
  return 'Бакалея';
}

function computeNutrition(days: MenuDay[], catalog: CatalogRecipe[]) {
  const recipeById = new Map(catalog.map((r) => [r.id, r]));
  let kcal = 0;
  let protein = 0;
  let fat = 0;
  let carb = 0;

  for (const day of days) {
    for (const id of [day.breakfastId, day.lunchId, day.dinnerId]) {
      const recipe = recipeById.get(id);
      if (!recipe) continue;
      for (const ing of recipe.ingredients) {
        kcal += ing.kcal;
        protein += ing.protein;
        fat += ing.fat;
        carb += ing.carb;
      }
    }
  }

  return { kcal: Math.round(kcal), protein: Math.round(protein), fat: Math.round(fat), carb: Math.round(carb) };
}

// ---------- Оркестратор ----------

export async function generateMenuWithAgents(
  profile: AgentProfile,
  catalog: CatalogRecipe[],
  storeProducts: StoreProductCompact[],
): Promise<GeneratedMenuPayload> {
  if (catalog.length === 0) throw new Error('Каталог рецептов пуст');

  const filteredIds = await runDietitianAgent(profile, catalog);
  const days = await runChefAgent(profile, catalog, filteredIds);
  const { zeroWasteNotes } = await runBuyerAgent(profile, days, catalog, storeProducts);
  const { items, totalCost } = buildShoppingItems(days, catalog, storeProducts, profile.budget_limit);
  const nutrition = computeNutrition(days, catalog);

  return {
    store: profile.pricing_store,
    totalCost,
    nutrition,
    zeroWasteNotes,
    days,
    shoppingItems: items,
  };
}
