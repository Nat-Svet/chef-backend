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
  /** Сколько порций (человек) нужно купить; по умолчанию 1. */
  portions?: number;
};

export type MenuDay = { day: WeekDay; breakfastId: number; lunchId: number; dinnerId: number };
export type ShoppingItem = {
  name: string;
  /** Сколько граммов реально нужно на неделю. */
  grams: number;
  category: ShoppingCategory;
  /** Цена целых упаковок: packs × цена упаковки. */
  price: number;
  store: string;
  /** Число целых упаковок к покупке (0 — товар не найден в каталоге, цена оценочная). */
  packs: number;
  /** Вес одной упаковки, г. */
  packGrams: number;
  packTitle: string;
};

export type GeneratedMenuPayload = {
  store: string;
  stores: string[];
  totalCost: number;
  nutrition: { kcal: number; protein: number; fat: number; carb: number } | null;
  zeroWasteNotes: string;
  scarcityNotice: string | null;
  /** Заполняется, только если при заданном бюджете и фильтрах уложиться в лимит невозможно. */
  budgetNotice: string | null;
  days: MenuDay[];
  shoppingItems: ShoppingItem[];
};

const WEEK_DAYS: WeekDay[] = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'];
/** Снятый с продукта тег: рецепты с ним в меню не попадают (см. server.js). */
const EQUIPMENT = [
  'Плита',
  'Духовка',
  'Мультиварка',
  'Микроволновка',
  'Блендер / Миксер',
  'Электрогриль / Аэрогриль',
  'Тостер',
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
// (planWeek повторяет уже отобранные блюда, а не ослабляет фильтр).
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

export type ChefResult = {
  days: MenuDay[];
  scarcity: { isScarce: boolean; availableCount: number };
  /** Полные (до сужения по бюджету) пулы слотов — для подгонки под лимит. */
  pools: Record<MealType, CatalogRecipe[]>;
};

async function runChefAgent(
  profile: AgentProfile,
  catalog: CatalogRecipe[],
  allowedIds: number[],
  pricer: Pricer,
): Promise<ChefResult> {
  const fullPools = buildMealPools(catalog, allowedIds, profile.equipment_tags);
  const scarcity = computeScarcity(fullPools);
  // Бюджет от обратного: потолок цены порции в слоте = бюджет / (21 × человек).
  const slotBudget = slotBudgetFor(profile.budget_limit, Math.max(1, Math.round(profile.portions ?? 1)));
  const mealPools = {} as Record<MealType, CatalogRecipe[]>;
  for (const type of MEAL_TYPES) {
    mealPools[type] = profile.budget_limit > 0 ? narrowPoolByBudget(fullPools[type], pricer, slotBudget) : fullPools[type];
  }
  const compact = MEAL_TYPES.flatMap((type) =>
    mealPools[type].map((r) => ({ id: r.id, meal_type: r.meal_type, tags: r.tags, cooking_time: r.cooking_time })),
  );

  const system = `Ты — Агент-Шеф приложения «Шеф в Кармане».
Составь меню на 7 дней (Пн..Вс) по 3 приёма пищи: завтрак, обед, ужин. Итого ровно 21 слот.
Правила:
1. Используй ТОЛЬКО id из переданного списка recipes, не выдумывай новые и не бери id другого meal_type.
2. meal_type блюда обязан соответствовать слоту (завтрак/обед/ужин).
3. Все 21 блюда недели должны быть РАЗНЫМИ id — не повторяй рецепт.
4. В рамках ОДНОГО дня не бери блюда с одинаковым главным белком (курица, говядина, рыба, яйца…) или одинаковым гарниром (гречка, рис, макароны, картофель…).
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
    return { days: planWeek(result.days, mealPools, catalog), scarcity, pools: fullPools };
  } catch {
    return { days: planWeek([], mealPools, catalog), scarcity, pools: fullPools };
  }
}

/** Блюдо подходит, если у пользователя есть ВСЯ нужная ему техника (рецепт
 *  может требовать сразу плиту и блендер). */
function isEquipmentCompatible(recipe: CatalogRecipe, equipmentTags: string[]): boolean {
  const required = recipe.tags.filter((tag) => EQUIPMENT.includes(tag));
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

// ---------- Гастрономическое разнообразие дня ----------

type DishInfo = { protein: string | null; side: string | null };

const PROTEIN_RULES: [RegExp, string][] = [
  [/яйц/i, 'яйца'],
  [/куриное филе|куриные|куриный/i, 'курица'],
  [/говя/i, 'говядина'],
  [/свинин/i, 'свинина'],
  [/индейк/i, 'индейка'],
  [/лосос|сёмг|треск|минтай|тунец/i, 'рыба'],
  [/креветк|кальмар/i, 'морепродукты'],
  [/тофу|нут|фасол|чечевиц/i, 'бобовые'],
  [/творог/i, 'творог'],
];

const SIDE_RULES: [RegExp, string][] = [
  [/гречк/i, 'гречка'],
  [/рис/i, 'рис'],
  [/макарон/i, 'макароны'],
  [/картофел/i, 'картофель'],
  [/булгур/i, 'булгур'],
  [/кускус/i, 'кускус'],
  [/киноа/i, 'киноа'],
  [/хлеб|хлебцы|лаваш/i, 'хлеб'],
  [/овсян/i, 'овсянка'],
  [/манк/i, 'манка'],
];

/** Главный белок = группа с наибольшим вкладом белка (от 10 г на порцию);
 *  гарнир = самая тяжёлая крахмальная позиция (от 40 г). */
export function dishInfo(recipe: CatalogRecipe): DishInfo {
  const proteinByGroup = new Map<string, number>();
  let side: string | null = null;
  let sideGrams = 0;

  for (const ing of recipe.ingredients) {
    const proteinRule = PROTEIN_RULES.find(([re]) => re.test(ing.name));
    if (proteinRule) {
      proteinByGroup.set(proteinRule[1], (proteinByGroup.get(proteinRule[1]) ?? 0) + (ing.protein || 0));
    }
    const sideRule = SIDE_RULES.find(([re]) => re.test(ing.name));
    if (sideRule && ing.grams >= 40 && ing.grams > sideGrams) {
      side = sideRule[1];
      sideGrams = ing.grams;
    }
  }

  let protein: string | null = null;
  let best = 10;
  for (const [group, value] of proteinByGroup) {
    if (value >= best) {
      protein = group;
      best = value;
    }
  }
  return { protein, side };
}

/** Два блюда одного дня "конфликтуют", если у них одинаковый главный белок или гарнир. */
export function clash(a: DishInfo | undefined, b: DishInfo | undefined): boolean {
  if (!a || !b) return false;
  return Boolean((a.protein && a.protein === b.protein) || (a.side && a.side === b.side));
}

function shuffle<T>(items: T[]): T[] {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

/** Строгий план недели: 21 разное блюдо, в каждом дне — разные главные белки и
 *  гарниры (перебор с возвратом, подсказки LLM пробуются первыми). Если пулы
 *  слишком малы для идеального плана — жадный запасной вариант, который
 *  минимизирует повторы рецептов, а затем конфликты внутри дня. */
function planWeek(rawDays: unknown, mealPools: Record<MealType, CatalogRecipe[]>, catalog: CatalogRecipe[]): MenuDay[] {
  const info = new Map(catalog.map((r) => [r.id, dishInfo(r)]));
  const poolIds = {} as Record<MealType, number[]>;
  for (const type of MEAL_TYPES) {
    const own = mealPools[type].length ? mealPools[type] : catalog.filter((r) => r.meal_type === type);
    if (!mealPools[type].length) console.warn(`Нет рецептов meal_type="${type}" под фильтры — берём весь каталог типа`);
    poolIds[type] = (own.length ? own : catalog).map((r) => r.id);
  }

  const source = Array.isArray(rawDays) ? (rawDays as any[]) : [];
  const hint = (type: MealType, day: number): number | undefined => {
    const row = source.find((item) => item?.day === WEEK_DAYS[day]) ?? source[day];
    const value = Number(type === 'завтрак' ? row?.breakfastId : type === 'обед' ? row?.lunchId : row?.dinnerId);
    return poolIds[type].includes(value) ? value : undefined;
  };

  const strictPlan = (attempt: number): number[][] | null => {
    const used = new Set<number>();
    const plan: number[][] = [];
    let work = 0;
    const order = MEAL_TYPES.map((type) => (attempt === 0 ? poolIds[type] : shuffle(poolIds[type])));

    const candidates = (typeIndex: number, day: number) => {
      const free = order[typeIndex].filter((id) => !used.has(id));
      const preferred = attempt === 0 ? hint(MEAL_TYPES[typeIndex], day) : undefined;
      return preferred !== undefined && free.includes(preferred)
        ? [preferred, ...free.filter((id) => id !== preferred)]
        : free;
    };

    const search = (day: number): boolean => {
      if (day === 7) return true;
      const breakfasts = candidates(0, day);
      const lunches = candidates(1, day);
      const dinners = candidates(2, day);
      for (const b of breakfasts) {
        for (const l of lunches) {
          if (++work > 150000) return false;
          if (clash(info.get(b), info.get(l))) continue;
          for (const n of dinners) {
            if (clash(info.get(b), info.get(n)) || clash(info.get(l), info.get(n))) continue;
            used.add(b);
            used.add(l);
            used.add(n);
            plan[day] = [b, l, n];
            if (search(day + 1)) return true;
            used.delete(b);
            used.delete(l);
            used.delete(n);
            if (work > 150000) return false;
          }
        }
      }
      return false;
    };

    return search(0) ? plan : null;
  };

  let plan: number[][] | null = null;
  if (MEAL_TYPES.every((type) => poolIds[type].length >= 7)) {
    for (let attempt = 0; attempt < 30 && !plan; attempt += 1) plan = strictPlan(attempt);
  }

  if (!plan) {
    const useCount = new Map<number, number>();
    const uses = (id: number) => useCount.get(id) ?? 0;
    plan = [];
    for (let day = 0; day < 7; day += 1) {
      let best: number[] = [poolIds.завтрак[0], poolIds.обед[0], poolIds.ужин[0]];
      let bestScore = Infinity;
      for (const b of poolIds.завтрак) {
        for (const l of poolIds.обед) {
          for (const n of poolIds.ужин) {
            const reused = (uses(b) > 0 ? 1 : 0) + (uses(l) > 0 ? 1 : 0) + (uses(n) > 0 ? 1 : 0);
            const clashes =
              (clash(info.get(b), info.get(l)) ? 1 : 0) +
              (clash(info.get(b), info.get(n)) ? 1 : 0) +
              (clash(info.get(l), info.get(n)) ? 1 : 0);
            const score = 100 * reused + uses(b) + uses(l) + uses(n) + 30 * clashes + Math.random();
            if (score < bestScore) {
              bestScore = score;
              best = [b, l, n];
            }
          }
        }
      }
      best.forEach((id) => useCount.set(id, uses(id) + 1));
      plan.push(best);
    }
  }

  return WEEK_DAYS.map((day, index) => ({
    day,
    breakfastId: plan![index][0],
    lunchId: plan![index][1],
    dinnerId: plan![index][2],
  }));
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

// ---------- Корзина из целых упаковок ----------

type Basket = {
  items: ShoppingItem[];
  /** Реальная стоимость: целые упаковки × цена упаковки. */
  totalCost: number;
  /** Пропорциональная оценка (граммы × цена за грамм) — только для сравнения вариантов. */
  proportionalCost: number;
};

type Pricer = {
  basket: (ids: number[], portions: number) => Basket;
  /** Оценка стоимости одной порции блюда по цене за грамм (без округления до упаковок). */
  portionCost: (recipe: CatalogRecipe) => number;
};

function perGramPrice(p: StoreProductCompact): number {
  return Number(p.price) / (p.pack_g && p.pack_g > 0 ? p.pack_g : 1);
}

/** Среди совпадений по названию берём самую дешёвую за грамм упаковку. */
function matchProduct(name: string, storeProducts: StoreProductCompact[]): StoreProductCompact | undefined {
  const needle = name.toLowerCase();
  const candidates = storeProducts.filter((p) => {
    const term = p.search_term.toLowerCase();
    return term === needle || needle.includes(term) || term.includes(needle);
  });
  if (!candidates.length) return undefined;
  return candidates.reduce((best, cur) => (perGramPrice(cur) < perGramPrice(best) ? cur : best));
}

function createPricer(catalog: CatalogRecipe[], storeProducts: StoreProductCompact[], fallbackStore: string): Pricer {
  const recipeById = new Map(catalog.map((r) => [r.id, r]));
  const productCache = new Map<string, StoreProductCompact | undefined>();
  const productFor = (name: string) => {
    if (!productCache.has(name)) productCache.set(name, matchProduct(name, storeProducts));
    return productCache.get(name);
  };

  const basket = (ids: number[], portions: number): Basket => {
    const counts = new Map<number, number>();
    for (const id of ids) counts.set(id, (counts.get(id) ?? 0) + 1);

    // Граммы суммируются по товару магазина: два ингредиента с одной и той же
    // упаковкой делят её, а не покупают по пачке каждый.
    type Line = { names: string[]; grams: number; product?: StoreProductCompact };
    const lines = new Map<string, Line>();
    for (const [id, times] of counts) {
      const recipe = recipeById.get(id);
      if (!recipe) continue;
      for (const ing of recipe.ingredients) {
        const product = productFor(ing.name);
        const key = product ? `p:${product.title}` : `n:${ing.name}`;
        const line = lines.get(key) ?? { names: [], grams: 0, product };
        if (!line.names.includes(ing.name)) line.names.push(ing.name);
        line.grams += ing.grams * times * portions;
        lines.set(key, line);
      }
    }

    let proportionalCost = 0;
    const items: ShoppingItem[] = [...lines.values()].map(({ names, grams, product }) => {
      const name = names.length === 1 ? names[0] : (product?.search_term ?? names[0]);
      if (!product) {
        const price = Math.max(1, Math.round(grams * 0.15));
        proportionalCost += price;
        return { name, grams, category: guessCategory(name), price, store: fallbackStore, packs: 0, packGrams: 0, packTitle: '' };
      }
      const packGrams = product.pack_g && product.pack_g > 0 ? product.pack_g : grams;
      const packs = Math.max(1, Math.ceil(grams / packGrams - 1e-9));
      proportionalCost += grams * perGramPrice(product);
      return {
        name,
        grams,
        category: guessCategory(name),
        price: Math.round(packs * Number(product.price)),
        store: product.store,
        packs,
        packGrams,
        packTitle: product.title,
      };
    });

    items.sort((x, y) => x.name.localeCompare(y.name, 'ru'));
    return { items, totalCost: items.reduce((sum, item) => sum + item.price, 0), proportionalCost };
  };

  const portionCost = (recipe: CatalogRecipe) =>
    recipe.ingredients.reduce((sum, ing) => {
      const product = productFor(ing.name);
      return sum + (product ? ing.grams * perGramPrice(product) : ing.grams * 0.15);
    }, 0);

  return { basket, portionCost };
}

// ---------- Бюджет: недельный лимит — жёсткий ----------

function daysToIds(days: MenuDay[]): number[] {
  return days.flatMap((d) => [d.breakfastId, d.lunchId, d.dinnerId]);
}

function idsToDays(ids: number[]): MenuDay[] {
  return WEEK_DAYS.map((day, i) => ({
    day,
    breakfastId: ids[i * 3],
    lunchId: ids[i * 3 + 1],
    dinnerId: ids[i * 3 + 2],
  }));
}

/** Бюджет «от обратного»: бюджет / (21 слот × число человек) — потолок стоимости одной порции в слоте. */
export function slotBudgetFor(budget: number, portions: number): number {
  return budget / (21 * Math.max(1, portions));
}

/** Сужает пул слота до блюд, чья порция укладывается в слотовый бюджет (с запасом на общие упаковки).
 *  Если таких меньше семи — берём семь самых дешёвых, чтобы неделя осталась без повторов. */
function narrowPoolByBudget(pool: CatalogRecipe[], pricer: Pricer, slotBudget: number): CatalogRecipe[] {
  const priced = pool.map((r) => ({ r, cost: pricer.portionCost(r) })).sort((a, b) => a.cost - b.cost);
  const affordable = priced.filter((p) => p.cost <= slotBudget * 1.3);
  const chosen = affordable.length >= 7 ? affordable : priced.slice(0, Math.min(7, priced.length));
  return chosen.map((p) => p.r);
}

/** Подгоняет неделю под лимит: пока реальная стоимость (целые упаковки) выше бюджета,
 *  делает лучшую замену одного блюда на более дешёвое из полного пула слота, не ломая
 *  разнообразие дня. Сначала без повторов блюд за неделю, затем — если иначе никак — с повторами. */
function fitMenuToBudget(
  days: MenuDay[],
  pools: Record<MealType, CatalogRecipe[]>,
  catalog: CatalogRecipe[],
  pricer: Pricer,
  budget: number,
  portions: number,
): { days: MenuDay[]; totalCost: number } {
  const info = new Map(catalog.map((r) => [r.id, dishInfo(r)]));
  const ids = daysToIds(days);
  const evaluate = (list: number[]) => {
    const b = pricer.basket(list, portions);
    return { total: b.totalCost, score: b.totalCost + 0.05 * b.proportionalCost };
  };

  let current = evaluate(ids);

  for (const allowRepeats of [false, true]) {
    for (let iter = 0; iter < 80 && current.total > budget; iter += 1) {
      let bestMove: { slot: number; id: number } | null = null;
      let bestEval = current;

      for (let slot = 0; slot < 21; slot += 1) {
        const type = MEAL_TYPES[slot % 3];
        const dayStart = Math.floor(slot / 3) * 3;
        const mates = [0, 1, 2].filter((k) => dayStart + k !== slot).map((k) => dayStart + k);
        for (const candidate of pools[type]) {
          if (candidate.id === ids[slot]) continue;
          if (!allowRepeats && ids.includes(candidate.id)) continue;
          if (mates.some((m) => clash(info.get(candidate.id), info.get(ids[m])))) continue;

          const previous = ids[slot];
          ids[slot] = candidate.id;
          const next = evaluate(ids);
          ids[slot] = previous;
          if (next.score < bestEval.score - 1e-6) {
            bestEval = next;
            bestMove = { slot, id: candidate.id };
          }
        }
      }

      if (!bestMove) break;
      ids[bestMove.slot] = bestMove.id;
      current = bestEval;
    }
    if (current.total <= budget) break;
  }

  return { days: idsToDays(ids), totalCost: current.total };
}

function buildBudgetNotice(budget: number, minCost: number, portions: number): string {
  const people = portions === 1 ? 'одного человека' : `${portions} чел.`;
  return `На ${people} и ваши настройки мы подобрали самое выгодное меню — ${minCost.toLocaleString('ru-RU')} ₽. Это больше лимита ${budget.toLocaleString('ru-RU')} ₽: чтобы уложиться, увеличьте бюджет или смягчите фильтры.`;
}

/** Для кнопки «Заменить блюдо»: из кандидатов оставляем те, с которыми вся неделя остаётся в бюджете;
 *  если таких нет — самые дешёвые по итоговому чеку. */
function filterByWeekBudget(
  candidates: CatalogRecipe[],
  ctx: { menuIds: number[]; slotIndex: number; portions: number; budget: number; pricer: Pricer },
): CatalogRecipe[] {
  const costWith = (id: number) => {
    const ids = [...ctx.menuIds];
    ids[ctx.slotIndex] = id;
    return ctx.pricer.basket(ids, ctx.portions).totalCost;
  };
  const costed = candidates.map((r) => ({ r, cost: costWith(r.id) }));
  const within = costed.filter((c) => c.cost <= ctx.budget);
  if (within.length) return within.map((c) => c.r);
  const cheapest = Math.min(...costed.map((c) => c.cost));
  return costed.filter((c) => c.cost === cheapest).map((c) => c.r);
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

  const portions = Math.max(1, Math.round(profile.portions ?? 1));
  const pricer = createPricer(catalog, storeProducts, profile.pricing_store);

  const filteredIds = await runDietitianAgent(profile, catalog);
  const chef = await runChefAgent(profile, catalog, filteredIds, pricer);
  const scarcity = chef.scarcity;

  // Закупщик: стоимость — целые упаковки; недельный бюджет — жёсткий лимит.
  const fitted =
    profile.budget_limit > 0
      ? fitMenuToBudget(chef.days, chef.pools, catalog, pricer, profile.budget_limit, portions)
      : { days: chef.days, totalCost: pricer.basket(daysToIds(chef.days), portions).totalCost };
  const days = fitted.days;
  const { zeroWasteNotes } = await runBuyerAgent(profile, days, catalog, storeProducts);
  const { items, totalCost } = pricer.basket(daysToIds(days), portions);
  const budgetNotice =
    profile.budget_limit > 0 && totalCost > profile.budget_limit
      ? buildBudgetNotice(profile.budget_limit, totalCost, portions)
      : null;
  const nutrition = computeNutrition(days, catalog);
  const stores = [...new Set(items.map((item) => item.store))];

  return {
    store: profile.pricing_store,
    stores: stores.length ? stores : [profile.pricing_store],
    totalCost,
    nutrition,
    zeroWasteNotes,
    scarcityNotice: scarcity.isScarce ? buildScarcityNotice(scarcity.availableCount) : null,
    budgetNotice,
    days,
    shoppingItems: items,
  };
}

/** Для кнопки «Изменить блюдо». Уровни выбора (первый непустой):
 *  1) не использовано на неделе и не отвергнуто пользователем;
 *  2) повтор из этой недели, но не отвергнутое;
 *  3) пул исчерпан — можно вернуть и отвергнутое.
 *  Внутри уровня предпочитаем блюда, не конфликтующие с соседями дня
 *  (siblingIds) по белку/гарниру. Фильтры diet/equipment строгие. */
export function pickReplacementRecipe(
  profile: AgentProfile,
  catalog: CatalogRecipe[],
  mealType: MealType,
  usedIds: number[],
  rejectedIds: number[] = [],
  siblingIds: number[] = [],
  budgetCtx?: { menuIds: number[]; slotIndex: number; portions: number; storeProducts: StoreProductCompact[] },
): number | null {
  const allowedIds = new Set(filterByDiet(catalog, profile.diet_tags));
  const byDiet = catalog.filter((r) => r.meal_type === mealType && allowedIds.has(r.id));
  const byEquip = byDiet.filter((r) => isEquipmentCompatible(r, profile.equipment_tags));
  const pool = byEquip.length ? byEquip : byDiet;
  if (!pool.length) return null;

  const used = new Set(usedIds);
  const rejected = new Set(rejectedIds);
  const siblings = siblingIds
    .map((id) => catalog.find((r) => r.id === id))
    .filter((r): r is CatalogRecipe => Boolean(r))
    .map(dishInfo);
  const diverse = (r: CatalogRecipe) => !siblings.some((s) => clash(dishInfo(r), s));
  const pickOne = (items: CatalogRecipe[]) => items[Math.floor(Math.random() * items.length)].id;

  // Недельный бюджет — жёсткий лимит и для замены: берём только блюда, с которыми чек недели не выходит за него.
  const affordable =
    budgetCtx && profile.budget_limit > 0 && budgetCtx.menuIds.length === 21
      ? new Set(
          filterByWeekBudget(pool, {
            menuIds: budgetCtx.menuIds,
            slotIndex: budgetCtx.slotIndex,
            portions: Math.max(1, Math.round(budgetCtx.portions)),
            budget: profile.budget_limit,
            pricer: createPricer(catalog, budgetCtx.storeProducts, profile.pricing_store),
          }).map((r) => r.id),
        )
      : null;
  const budgetPool = affordable ? pool.filter((r) => affordable.has(r.id)) : pool;

  const tiers = [
    budgetPool.filter((r) => !used.has(r.id) && !rejected.has(r.id)),
    budgetPool.filter((r) => !rejected.has(r.id)),
    budgetPool,
  ];

  for (const tier of tiers) {
    const good = tier.filter(diverse);
    if (good.length) return pickOne(good);
  }
  for (const tier of tiers) {
    if (tier.length) return pickOne(tier);
  }
  return null;
}
