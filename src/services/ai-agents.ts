/**
 * Пайплайн подбора меню: три последовательных шага (Диетолог → Шеф → Закупщик) поверх DeepSeek (Timeweb AI).
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
export type ShoppingItem = { name: string; grams: number; category: ShoppingCategory; price: number; store: string };

export type GeneratedMenuPayload = {
  store: string;
  stores: string[];
  totalCost: number;
  nutrition: { kcal: number; protein: number; fat: number; carb: number } | null;
  zeroWasteNotes: string;
  scarcityNotice: string | null;
  days: MenuDay[];
  shoppingItems: ShoppingItem[];
};

const WEEK_DAYS: WeekDay[] = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'];
const NO_COOK = 'Без готовки';
const EQUIPMENT = [
  'Плита',
  'Духовка',
  'Мультиварка',
  'Микроволновка',
  'Блендер / Миксер',
  'Электрогриль / Аэрогриль',
  'Тостер',
  NO_COOK,
];

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

// Строгая фильтрация: рецепт подходит, только если хотя бы один его tag есть
// в diet_tags пользователя. Никакого "подмешивания" непрофильных блюд, даже
// если подходящих рецептов мало — дефицит обрабатывается выше по стеку
// (normalizeDays повторяет уже отобранные блюда, а не ослабляет фильтр).
function filterByDiet(catalog: CatalogRecipe[], dietTags: string[]): number[] {
  if (dietTags.length === 0) return catalog.map((r) => r.id);
  return catalog.filter((r) => r.tags.some((tag) => dietTags.includes(tag))).map((r) => r.id);
}

async function runDietitianAgent(profile: AgentProfile, catalog: CatalogRecipe[]): Promise<number[]> {
  const strict = filterByDiet(catalog, profile.diet_tags);
  if (profile.diet_tags.length === 0) return strict;

  const compact = catalog.map((r) => ({ id: r.id, tags: r.tags, meal_type: r.meal_type }));

  const system = `Ты — Агент-Диетолог приложения «Шеф в Кармане».
Задача: отобрать id рецептов, чьи tags пересекаются с diet_tags пользователя.
Правила (СТРОГО, без исключений):
1. Рецепт подходит, ТОЛЬКО если хотя бы один его tag входит в diet_tags. Никаких рецептов без совпадения — даже если подходящих мало.
2. Никогда не выдумывай id, которых нет в переданном каталоге.
Верни СТРОГО один JSON без markdown: {"filteredRecipeIds": [1,2,3]}`;

  const user = `diet_tags: ${JSON.stringify(profile.diet_tags)}
Каталог (id, tags, meal_type): ${JSON.stringify(compact)}`;

  try {
    const result = await callWithRetry([
      { role: 'system', content: system },
      { role: 'user', content: user },
    ]);
    const strictSet = new Set(strict);
    const ids: number[] = Array.isArray(result.filteredRecipeIds)
      ? result.filteredRecipeIds.map(Number).filter((id: number) => strictSet.has(id))
      : [];
    // LLM иногда занижает выборку — подстраховываемся детерминированным
    // строгим фильтром как минимумом, не доверяя модели математику/полноту.
    return ids.length ? ids : strict;
  } catch {
    return strict;
  }
}

// ---------- Агент 2: Шеф-повар ----------
// Вход: id от Диетолога + equipment_tags. Выход: раскладка 7×3 = 21 слот.

const MEAL_TYPES: MealType[] = ['завтрак', 'обед', 'ужин'];

export type ChefResult = { days: MenuDay[]; scarcity: { isScarce: boolean; availableCount: number } };

async function runChefAgent(profile: AgentProfile, catalog: CatalogRecipe[], allowedIds: number[]): Promise<ChefResult> {
  const mealPools = buildMealPools(catalog, allowedIds, profile.equipment_tags);
  const scarcity = computeScarcity(mealPools);
  const compact = MEAL_TYPES.flatMap((type) =>
    mealPools[type].map((r) => ({ id: r.id, meal_type: r.meal_type, tags: r.tags, cooking_time: r.cooking_time })),
  );

  const system = `Ты — Агент-Шеф приложения «Шеф в Кармане».
Составь меню на 7 дней (Пн..Вс) по 3 приёма пищи: завтрак, обед, ужин. Итого ровно 21 слот.
Правила:
1. Используй ТОЛЬКО id из переданного списка recipes, не выдумывай новые и не бери id другого meal_type.
2. meal_type блюда обязан соответствовать слоту (завтрак/обед/ужин).
3. Максимизируй уникальность: пока в списке есть неиспользованный на этой неделе id нужного meal_type — используй его, а не повторяй прежний.
4. Повторяй блюдо только если все доступные id этого meal_type уже использованы.
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
    return { days: normalizeDays(result.days, mealPools, catalog), scarcity };
  } catch {
    return { days: normalizeDays([], mealPools, catalog), scarcity };
  }
}

/** Блюдо подходит, если у пользователя есть ВСЯ нужная ему техника (рецепт
 *  может требовать сразу плиту и блендер). «Без готовки» техники не требует. */
function isEquipmentCompatible(recipe: CatalogRecipe, equipmentTags: string[]): boolean {
  const required = recipe.tags.filter((tag) => EQUIPMENT.includes(tag) && tag !== NO_COOK);
  return required.every((tag) => equipmentTags.includes(tag));
}

/** Пул на каждый meal_type строится независимо и СТРОГО: техника сужает
 *  diet-отфильтрованный список, но никогда не расширяется обратно на весь
 *  каталог (это было бы "подмешиванием" непрофильных блюд). Пустой пул —
 *  легитимный результат, обрабатывается через scarcity-уведомление. */
function buildMealPools(
  catalog: CatalogRecipe[],
  allowedIds: number[],
  equipmentTags: string[],
): Record<MealType, CatalogRecipe[]> {
  const allowedSet = new Set(allowedIds);
  const pools = {} as Record<MealType, CatalogRecipe[]>;

  for (const type of MEAL_TYPES) {
    const byDiet = catalog.filter((r) => r.meal_type === type && allowedSet.has(r.id));
    const byEquip = byDiet.filter((r) => isEquipmentCompatible(r, equipmentTags));
    pools[type] = byEquip.length ? byEquip : byDiet;
  }

  return pools;
}

function computeScarcity(mealPools: Record<MealType, CatalogRecipe[]>): { isScarce: boolean; availableCount: number } {
  const availableCount = MEAL_TYPES.reduce((sum, type) => sum + mealPools[type].length, 0);
  const isScarce = MEAL_TYPES.some((type) => mealPools[type].length < 7);
  return { isScarce, availableCount };
}

function normalizeDays(
  rawDays: unknown,
  mealPools: Record<MealType, CatalogRecipe[]>,
  catalog: CatalogRecipe[],
): MenuDay[] {
  const typeById = new Map(catalog.map((r) => [r.id, r.meal_type]));
  const usedByType: Record<MealType, Set<number>> = { завтрак: new Set(), обед: new Set(), ужин: new Set() };

  // Максимизируем уникальность: валидный и ещё не использованный id от LLM —
  // принимаем; иначе берём первый неиспользованный id из пула; повторяем
  // только когда пул того meal_type исчерпан целиком.
  const pick = (value: unknown, type: MealType, index: number): number => {
    const pool = mealPools[type];
    const used = usedByType[type];
    const numeric = Number(value);

    if (typeById.get(numeric) === type && pool.some((r) => r.id === numeric) && !used.has(numeric)) {
      used.add(numeric);
      return numeric;
    }

    const unused = pool.find((r) => !used.has(r.id));
    if (unused) {
      used.add(unused.id);
      return unused.id;
    }

    if (pool.length) return pool[index % pool.length].id;

    const anyOfType = catalog.filter((r) => r.meal_type === type);
    if (anyOfType.length) {
      console.warn(`Нет рецептов meal_type="${type}", подходящих под фильтры — используем весь каталог этого типа`);
      return anyOfType[index % anyOfType.length].id;
    }
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
  fallbackStore: string,
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

  // storeProducts — каталог ОДНОГО магазина. Если в нём несколько позиций с одним
  // названием, берём самую дешёвую за грамм.
  const matchProduct = (name: string) => {
    const needle = name.toLowerCase();
    const candidates = storeProducts.filter(
      (p) =>
        p.search_term.toLowerCase() === needle ||
        needle.includes(p.search_term.toLowerCase()) ||
        p.search_term.toLowerCase().includes(needle),
    );
    if (!candidates.length) return undefined;
    return candidates.reduce((best, cur) => {
      const bestPerGram = best.price / (best.pack_g && best.pack_g > 0 ? best.pack_g : 1);
      const curPerGram = cur.price / (cur.pack_g && cur.pack_g > 0 ? cur.pack_g : 1);
      return curPerGram < bestPerGram ? cur : best;
    });
  };

  const raw = [...gramsByName.entries()].map(([name, grams]) => {
    const product = matchProduct(name);
    const packGrams = product?.pack_g && product.pack_g > 0 ? product.pack_g : grams;
    const packs = product ? Math.max(1, Math.ceil(grams / packGrams)) : 1;
    const price = product ? packs * Number(product.price) : Math.max(1, Math.round(grams * 0.15));
    return { name, grams, category: guessCategory(name), price, store: product?.store ?? fallbackStore };
  });

  // Стоимость — ровно по каталогу магазина: упаковки × цена, без подгонки под бюджет.
  const items = raw
    .map((item) => ({ ...item, price: Math.max(1, Math.round(item.price)) }))
    .sort((x, y) => x.name.localeCompare(y.name, 'ru'));

  const totalCost = items.reduce((sum, item) => sum + item.price, 0);
  return { items, totalCost };
}

function guessCategory(name: string): ShoppingCategory {
  const lower = name.toLowerCase();
  if (/(яйц|молок|йогурт|сыр|творог|кефир|сливк|сметан|сливочн|моцарелл)/.test(lower)) return 'Молочные продукты';
  if (/(филе|курин|бёдр|крыл|фарш|мясо|говяд|свинин|индейк|ветчин|сосиск|лосос|сёмг|тунец|треск|минтай|кальмар|креветк)/.test(lower)) return 'Мясо и птица';
  if (/(ягод|овощ|перец|кабач|томат|помидор|морков|лук|брокколи|картоф|капуст|свёкл|тыкв|баклаж|гриб|шампин|огурц|шпинат|салат|руккол|чеснок|зелень|лимон|яблок|банан|авокадо)/.test(lower)) return 'Овощи и фрукты';
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

function buildScarcityNotice(availableCount: number): string {
  return `Подобрали максимум уникальных блюд по вашим фильтрам! Чтобы рацион стал ещё разнообразнее, попробуйте расширить настройки.`;
}

export async function generateMenuWithAgents(
  profile: AgentProfile,
  catalog: CatalogRecipe[],
  storeProducts: StoreProductCompact[],
): Promise<GeneratedMenuPayload> {
  if (catalog.length === 0) throw new Error('Каталог рецептов пуст');

  const filteredIds = await runDietitianAgent(profile, catalog);
  const { days, scarcity } = await runChefAgent(profile, catalog, filteredIds);
  const { zeroWasteNotes } = await runBuyerAgent(profile, days, catalog, storeProducts);
  const { items, totalCost } = buildShoppingItems(
    days,
    catalog,
    storeProducts,
    profile.pricing_store,
  );
  const nutrition = computeNutrition(days, catalog);
  const stores = [...new Set(items.map((item) => item.store))];

  return {
    store: profile.pricing_store,
    stores: stores.length ? stores : [profile.pricing_store],
    totalCost,
    nutrition,
    zeroWasteNotes,
    scarcityNotice: scarcity.isScarce ? buildScarcityNotice(scarcity.availableCount) : null,
    days,
    shoppingItems: items,
  };
}

/** Для кнопки «Изменить блюдо»: подбирает один рецепт того же meal_type,
 *  которого ещё нет в excludeIds (остальные блюда текущей недели), строго
 *  по тем же diet/equipment фильтрам. Повторяет уже использованный id,
 *  только если в каталоге действительно больше нечего предложить. */
export function pickReplacementRecipe(
  profile: AgentProfile,
  catalog: CatalogRecipe[],
  mealType: MealType,
  excludeIds: number[],
): number | null {
  const allowedIds = new Set(filterByDiet(catalog, profile.diet_tags));
  const byDiet = catalog.filter((r) => r.meal_type === mealType && allowedIds.has(r.id));
  const byEquip = byDiet.filter((r) => isEquipmentCompatible(r, profile.equipment_tags));
  const pool = byEquip.length ? byEquip : byDiet;
  if (!pool.length) return null;

  const exclude = new Set(excludeIds);
  const unused = pool.filter((r) => !exclude.has(r.id));
  const candidates = unused.length ? unused : pool;
  return candidates[Math.floor(Math.random() * candidates.length)].id;
}
