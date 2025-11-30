// Comprehensive JavaScript stress test - 1000+ lines
// Tests various JS constructs that exercise the grammar extensively

// ============ PART 1: Class Declarations with Complex Inheritance ============

class EventEmitter {
    constructor() {
        this.events = new Map();
        this.maxListeners = 10;
    }

    on(event, listener) {
        if (!this.events.has(event)) {
            this.events.set(event, []);
        }
        const listeners = this.events.get(event);
        if (listeners.length >= this.maxListeners) {
            console.warn(`MaxListenersExceeded: ${event}`);
        }
        listeners.push(listener);
        return this;
    }

    emit(event, ...args) {
        const listeners = this.events.get(event);
        if (!listeners) return false;
        listeners.forEach(listener => listener.apply(this, args));
        return true;
    }

    off(event, listener) {
        const listeners = this.events.get(event);
        if (!listeners) return this;
        const index = listeners.indexOf(listener);
        if (index !== -1) {
            listeners.splice(index, 1);
        }
        return this;
    }

    once(event, listener) {
        const wrapper = (...args) => {
            this.off(event, wrapper);
            listener.apply(this, args);
        };
        return this.on(event, wrapper);
    }
}

class Logger extends EventEmitter {
    static LEVELS = {
        DEBUG: 0,
        INFO: 1,
        WARN: 2,
        ERROR: 3,
        FATAL: 4
    };

    constructor(options = {}) {
        super();
        this.level = options.level ?? Logger.LEVELS.INFO;
        this.prefix = options.prefix ?? '';
        this.timestamps = options.timestamps ?? true;
        this.buffer = [];
        this.maxBuffer = options.maxBuffer ?? 1000;
    }

    #formatMessage(level, message, ...args) {
        const timestamp = this.timestamps ? `[${new Date().toISOString()}]` : '';
        const prefix = this.prefix ? `[${this.prefix}]` : '';
        const levelName = Object.entries(Logger.LEVELS)
            .find(([_, v]) => v === level)?.[0] ?? 'UNKNOWN';
        return `${timestamp}${prefix}[${levelName}] ${message}`;
    }

    #log(level, message, ...args) {
        if (level < this.level) return;
        const formatted = this.#formatMessage(level, message, ...args);
        this.buffer.push({ level, message: formatted, timestamp: Date.now() });
        if (this.buffer.length > this.maxBuffer) {
            this.buffer.shift();
        }
        this.emit('log', { level, message: formatted });
        return formatted;
    }

    debug(message, ...args) { return this.#log(Logger.LEVELS.DEBUG, message, ...args); }
    info(message, ...args) { return this.#log(Logger.LEVELS.INFO, message, ...args); }
    warn(message, ...args) { return this.#log(Logger.LEVELS.WARN, message, ...args); }
    error(message, ...args) { return this.#log(Logger.LEVELS.ERROR, message, ...args); }
    fatal(message, ...args) { return this.#log(Logger.LEVELS.FATAL, message, ...args); }
}

// ============ PART 2: Complex Data Structures ============

class PriorityQueue {
    constructor(comparator = (a, b) => a - b) {
        this.heap = [];
        this.comparator = comparator;
    }

    get size() { return this.heap.length; }
    get isEmpty() { return this.size === 0; }

    #parent(i) { return Math.floor((i - 1) / 2); }
    #left(i) { return 2 * i + 1; }
    #right(i) { return 2 * i + 2; }

    #swap(i, j) {
        [this.heap[i], this.heap[j]] = [this.heap[j], this.heap[i]];
    }

    #siftUp(i) {
        while (i > 0 && this.comparator(this.heap[this.#parent(i)], this.heap[i]) > 0) {
            this.#swap(i, this.#parent(i));
            i = this.#parent(i);
        }
    }

    #siftDown(i) {
        let min = i;
        const left = this.#left(i);
        const right = this.#right(i);

        if (left < this.size && this.comparator(this.heap[left], this.heap[min]) < 0) {
            min = left;
        }
        if (right < this.size && this.comparator(this.heap[right], this.heap[min]) < 0) {
            min = right;
        }
        if (min !== i) {
            this.#swap(i, min);
            this.#siftDown(min);
        }
    }

    push(value) {
        this.heap.push(value);
        this.#siftUp(this.size - 1);
        return this.size;
    }

    pop() {
        if (this.isEmpty) return undefined;
        const result = this.heap[0];
        const last = this.heap.pop();
        if (!this.isEmpty) {
            this.heap[0] = last;
            this.#siftDown(0);
        }
        return result;
    }

    peek() { return this.heap[0]; }
}

class LRUCache {
    constructor(capacity) {
        this.capacity = capacity;
        this.cache = new Map();
    }

    get(key) {
        if (!this.cache.has(key)) return undefined;
        const value = this.cache.get(key);
        this.cache.delete(key);
        this.cache.set(key, value);
        return value;
    }

    put(key, value) {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        } else if (this.cache.size >= this.capacity) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
        this.cache.set(key, value);
    }
}

// ============ PART 3: Async/Await Patterns ============

async function fetchWithRetry(url, options = {}, maxRetries = 3) {
    const delays = [1000, 2000, 4000]; // Exponential backoff
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), options.timeout ?? 30000);
            
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });
            
            clearTimeout(timeout);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return response;
        } catch (error) {
            if (attempt === maxRetries) {
                throw error;
            }
            
            const delay = delays[Math.min(attempt, delays.length - 1)];
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
}

async function* paginate(fetchFn, pageSize = 10) {
    let page = 0;
    let hasMore = true;
    
    while (hasMore) {
        const { data, total } = await fetchFn({ page, pageSize });
        yield* data;
        page++;
        hasMore = page * pageSize < total;
    }
}

class AsyncQueue {
    constructor(concurrency = 1) {
        this.concurrency = concurrency;
        this.running = 0;
        this.queue = [];
    }

    async push(task) {
        return new Promise((resolve, reject) => {
            this.queue.push({ task, resolve, reject });
            this.#process();
        });
    }

    async #process() {
        if (this.running >= this.concurrency || this.queue.length === 0) {
            return;
        }

        this.running++;
        const { task, resolve, reject } = this.queue.shift();

        try {
            const result = await task();
            resolve(result);
        } catch (error) {
            reject(error);
        } finally {
            this.running--;
            this.#process();
        }
    }
}

// ============ PART 4: Functional Programming Patterns ============

const compose = (...fns) => (x) => fns.reduceRight((acc, fn) => fn(acc), x);
const pipe = (...fns) => (x) => fns.reduce((acc, fn) => fn(acc), x);

const curry = (fn) => {
    const arity = fn.length;
    return function curried(...args) {
        if (args.length >= arity) {
            return fn.apply(this, args);
        }
        return (...moreArgs) => curried.apply(this, [...args, ...moreArgs]);
    };
};

const memoize = (fn, getKey = (...args) => JSON.stringify(args)) => {
    const cache = new Map();
    return function memoized(...args) {
        const key = getKey.apply(this, args);
        if (cache.has(key)) {
            return cache.get(key);
        }
        const result = fn.apply(this, args);
        cache.set(key, result);
        return result;
    };
};

const debounce = (fn, wait, options = {}) => {
    let timeout;
    let lastArgs;
    let lastThis;
    let result;
    const { leading = false, trailing = true } = options;

    function debounced(...args) {
        lastArgs = args;
        lastThis = this;

        const invokeFunc = () => {
            result = fn.apply(lastThis, lastArgs);
            lastArgs = lastThis = undefined;
        };

        const shouldCallNow = leading && !timeout;

        clearTimeout(timeout);
        timeout = setTimeout(() => {
            timeout = undefined;
            if (trailing && lastArgs) {
                invokeFunc();
            }
        }, wait);

        if (shouldCallNow) {
            invokeFunc();
        }

        return result;
    }

    debounced.cancel = () => {
        clearTimeout(timeout);
        timeout = lastArgs = lastThis = undefined;
    };

    return debounced;
};

const throttle = (fn, wait, options = {}) => {
    let timeout;
    let previous = 0;
    const { leading = true, trailing = true } = options;

    function throttled(...args) {
        const now = Date.now();

        if (!previous && !leading) {
            previous = now;
        }

        const remaining = wait - (now - previous);

        if (remaining <= 0 || remaining > wait) {
            if (timeout) {
                clearTimeout(timeout);
                timeout = undefined;
            }
            previous = now;
            return fn.apply(this, args);
        }

        if (!timeout && trailing) {
            timeout = setTimeout(() => {
                previous = leading ? Date.now() : 0;
                timeout = undefined;
                fn.apply(this, args);
            }, remaining);
        }
    }

    throttled.cancel = () => {
        clearTimeout(timeout);
        previous = 0;
        timeout = undefined;
    };

    return throttled;
};

// ============ PART 5: Complex Object Manipulation ============

function deepClone(obj, seen = new WeakMap()) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }

    if (seen.has(obj)) {
        return seen.get(obj);
    }

    if (obj instanceof Date) {
        return new Date(obj.getTime());
    }

    if (obj instanceof RegExp) {
        return new RegExp(obj.source, obj.flags);
    }

    if (obj instanceof Map) {
        const clone = new Map();
        seen.set(obj, clone);
        obj.forEach((value, key) => {
            clone.set(deepClone(key, seen), deepClone(value, seen));
        });
        return clone;
    }

    if (obj instanceof Set) {
        const clone = new Set();
        seen.set(obj, clone);
        obj.forEach(value => {
            clone.add(deepClone(value, seen));
        });
        return clone;
    }

    if (Array.isArray(obj)) {
        const clone = [];
        seen.set(obj, clone);
        obj.forEach((item, index) => {
            clone[index] = deepClone(item, seen);
        });
        return clone;
    }

    const clone = Object.create(Object.getPrototypeOf(obj));
    seen.set(obj, clone);

    for (const key of Reflect.ownKeys(obj)) {
        const descriptor = Object.getOwnPropertyDescriptor(obj, key);
        if (descriptor.value !== undefined) {
            descriptor.value = deepClone(descriptor.value, seen);
        }
        Object.defineProperty(clone, key, descriptor);
    }

    return clone;
}

function deepMerge(target, ...sources) {
    if (!sources.length) return target;

    const source = sources.shift();

    if (isObject(target) && isObject(source)) {
        for (const key of Object.keys(source)) {
            if (isObject(source[key])) {
                if (!target[key]) {
                    Object.assign(target, { [key]: {} });
                }
                deepMerge(target[key], source[key]);
            } else {
                Object.assign(target, { [key]: source[key] });
            }
        }
    }

    return deepMerge(target, ...sources);
}

function isObject(item) {
    return item && typeof item === 'object' && !Array.isArray(item);
}

function flatten(arr, depth = Infinity) {
    if (depth < 1) return arr.slice();
    
    return arr.reduce((acc, val) => {
        if (Array.isArray(val)) {
            acc.push(...flatten(val, depth - 1));
        } else {
            acc.push(val);
        }
        return acc;
    }, []);
}

function groupBy(arr, keyFn) {
    return arr.reduce((groups, item) => {
        const key = keyFn(item);
        if (!groups[key]) {
            groups[key] = [];
        }
        groups[key].push(item);
        return groups;
    }, {});
}

// ============ PART 6: Iterator and Generator Patterns ============

function* range(start, end, step = 1) {
    if (step === 0) throw new Error('Step cannot be zero');
    
    if (step > 0) {
        for (let i = start; i < end; i += step) {
            yield i;
        }
    } else {
        for (let i = start; i > end; i += step) {
            yield i;
        }
    }
}

function* zip(...iterables) {
    const iterators = iterables.map(it => it[Symbol.iterator]());
    
    while (true) {
        const results = iterators.map(it => it.next());
        if (results.some(r => r.done)) {
            return;
        }
        yield results.map(r => r.value);
    }
}

function* enumerate(iterable, start = 0) {
    let index = start;
    for (const value of iterable) {
        yield [index++, value];
    }
}

function* take(iterable, n) {
    let count = 0;
    for (const value of iterable) {
        if (count >= n) return;
        yield value;
        count++;
    }
}

function* filter(iterable, predicate) {
    for (const value of iterable) {
        if (predicate(value)) {
            yield value;
        }
    }
}

function* map(iterable, transform) {
    for (const value of iterable) {
        yield transform(value);
    }
}

function reduce(iterable, reducer, initial) {
    let accumulator = initial;
    let isFirst = initial === undefined;
    
    for (const value of iterable) {
        if (isFirst) {
            accumulator = value;
            isFirst = false;
        } else {
            accumulator = reducer(accumulator, value);
        }
    }
    
    return accumulator;
}

// ============ PART 7: Complex Control Flow ============

function createStateMachine(config) {
    let currentState = config.initial;
    const listeners = [];

    return {
        get state() { return currentState; },

        transition(event, payload) {
            const stateConfig = config.states[currentState];
            if (!stateConfig || !stateConfig.on) return false;

            const transition = stateConfig.on[event];
            if (!transition) return false;

            const nextState = typeof transition === 'string'
                ? transition
                : transition.target;

            const previousState = currentState;
            currentState = nextState;

            if (typeof transition === 'object' && transition.action) {
                transition.action(payload);
            }

            listeners.forEach(listener => {
                listener({ from: previousState, to: currentState, event, payload });
            });

            return true;
        },

        subscribe(listener) {
            listeners.push(listener);
            return () => {
                const index = listeners.indexOf(listener);
                if (index !== -1) {
                    listeners.splice(index, 1);
                }
            };
        },

        can(event) {
            const stateConfig = config.states[currentState];
            return !!(stateConfig && stateConfig.on && stateConfig.on[event]);
        }
    };
}

// ============ PART 8: Complex String Operations ============

const StringUtils = {
    escapeHtml(str) {
        const htmlEntities = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        };
        return str.replace(/[&<>"']/g, char => htmlEntities[char]);
    },

    truncate(str, length, suffix = '...') {
        if (str.length <= length) return str;
        return str.slice(0, length - suffix.length) + suffix;
    },

    camelCase(str) {
        return str
            .replace(/(?:^\w|[A-Z]|\b\w)/g, (letter, index) =>
                index === 0 ? letter.toLowerCase() : letter.toUpperCase()
            )
            .replace(/\s+/g, '');
    },

    kebabCase(str) {
        return str
            .replace(/([a-z])([A-Z])/g, '$1-$2')
            .replace(/[\s_]+/g, '-')
            .toLowerCase();
    },

    snakeCase(str) {
        return str
            .replace(/([a-z])([A-Z])/g, '$1_$2')
            .replace(/[\s-]+/g, '_')
            .toLowerCase();
    },

    template(str, data) {
        return str.replace(/\{\{(\w+)\}\}/g, (match, key) => {
            return data.hasOwnProperty(key) ? data[key] : match;
        });
    },

    wordWrap(str, width, br = '\n') {
        const words = str.split(' ');
        const lines = [];
        let currentLine = '';

        for (const word of words) {
            if (currentLine.length + word.length + 1 <= width) {
                currentLine += (currentLine ? ' ' : '') + word;
            } else {
                if (currentLine) lines.push(currentLine);
                currentLine = word;
            }
        }

        if (currentLine) lines.push(currentLine);
        return lines.join(br);
    }
};

// ============ PART 9: Complex Number Operations ============

const MathUtils = {
    clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
    },

    lerp(start, end, t) {
        return start + (end - start) * t;
    },

    map(value, inMin, inMax, outMin, outMax) {
        return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
    },

    randomInt(min, max) {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    },

    gcd(a, b) {
        while (b !== 0) {
            [a, b] = [b, a % b];
        }
        return a;
    },

    lcm(a, b) {
        return (a * b) / this.gcd(a, b);
    },

    isPrime(n) {
        if (n < 2) return false;
        if (n === 2) return true;
        if (n % 2 === 0) return false;
        for (let i = 3; i <= Math.sqrt(n); i += 2) {
            if (n % i === 0) return false;
        }
        return true;
    },

    factorial(n) {
        if (n < 0) throw new Error('Factorial of negative number');
        if (n <= 1) return 1;
        return n * this.factorial(n - 1);
    },

    fibonacci(n) {
        if (n < 0) throw new Error('Fibonacci of negative index');
        if (n <= 1) return n;
        let [a, b] = [0, 1];
        for (let i = 2; i <= n; i++) {
            [a, b] = [b, a + b];
        }
        return b;
    }
};

// ============ PART 10: Complex Array Operations ============

const ArrayUtils = {
    chunk(arr, size) {
        const chunks = [];
        for (let i = 0; i < arr.length; i += size) {
            chunks.push(arr.slice(i, i + size));
        }
        return chunks;
    },

    shuffle(arr) {
        const shuffled = [...arr];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    },

    unique(arr) {
        return [...new Set(arr)];
    },

    intersection(arr1, arr2) {
        const set = new Set(arr2);
        return arr1.filter(item => set.has(item));
    },

    difference(arr1, arr2) {
        const set = new Set(arr2);
        return arr1.filter(item => !set.has(item));
    },

    union(arr1, arr2) {
        return [...new Set([...arr1, ...arr2])];
    },

    symmetricDifference(arr1, arr2) {
        const set1 = new Set(arr1);
        const set2 = new Set(arr2);
        return [
            ...arr1.filter(item => !set2.has(item)),
            ...arr2.filter(item => !set1.has(item))
        ];
    },

    sortBy(arr, keyFn) {
        return [...arr].sort((a, b) => {
            const keyA = keyFn(a);
            const keyB = keyFn(b);
            if (keyA < keyB) return -1;
            if (keyA > keyB) return 1;
            return 0;
        });
    },

    partition(arr, predicate) {
        const pass = [];
        const fail = [];
        for (const item of arr) {
            (predicate(item) ? pass : fail).push(item);
        }
        return [pass, fail];
    }
};

// ============ EXPORT ============

export {
    EventEmitter,
    Logger,
    PriorityQueue,
    LRUCache,
    AsyncQueue,
    fetchWithRetry,
    paginate,
    compose,
    pipe,
    curry,
    memoize,
    debounce,
    throttle,
    deepClone,
    deepMerge,
    flatten,
    groupBy,
    range,
    zip,
    enumerate,
    take,
    filter,
    map,
    reduce,
    createStateMachine,
    StringUtils,
    MathUtils,
    ArrayUtils
};
