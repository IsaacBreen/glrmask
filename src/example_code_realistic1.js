// Real-world JavaScript patterns and common use cases

// Module pattern with private variables
const UserManager = (function() {
    let users = [];
    let nextId = 1;

    function validateUser(user) {
        return user &&
               typeof user.name === 'string' &&
               user.name.trim().length > 0 &&
               typeof user.email === 'string' &&
               /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(user.email);
    }

    return {
        addUser: function(userData) {
            if (!validateUser(userData)) {
                throw new Error('Invalid user data');
            }

            const user = {
                id: nextId++,
                name: userData.name.trim(),
                email: userData.email.toLowerCase(),
                createdAt: new Date(),
                isActive: true
            };

            users.push(user);
            return user;
        },

        getUser: function(id) {
            return users.find(user => user.id === id);
        },

        getAllUsers: function() {
            return [...users]; // Return copy to prevent mutation
        },

        updateUser: function(id, updates) {
            const userIndex = users.findIndex(user => user.id === id);
            if (userIndex === -1) {
                throw new Error('User not found');
            }

            const updatedUser = {
                ...users[userIndex],
                ...updates,
                updatedAt: new Date()
            };

            if (!validateUser(updatedUser)) {
                throw new Error('Invalid update data');
            }

            users[userIndex] = updatedUser;
            return updatedUser;
        },

        deleteUser: function(id) {
            const userIndex = users.findIndex(user => user.id === id);
            if (userIndex === -1) {
                return false;
            }
            users.splice(userIndex, 1);
            return true;
        },

        searchUsers: function(query) {
            const searchTerm = query.toLowerCase();
            return users.filter(user =>
                user.name.toLowerCase().includes(searchTerm) ||
                user.email.includes(searchTerm)
            );
        },

        getUserCount: function() {
            return users.length;
        }
    };
})();

// API service class with error handling
class ApiService {
    constructor(baseURL, timeout = 5000) {
        this.baseURL = baseURL;
        this.timeout = timeout;
        this.cache = new Map();
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const cacheKey = `${endpoint}:${JSON.stringify(options)}`;

        // Check cache first
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            // Cache successful responses
            this.cache.set(cacheKey, data);

            return data;
        } catch (error) {
            clearTimeout(timeoutId);

            if (error.name === 'AbortError') {
                throw new Error('Request timeout');
            }

            throw error;
        }
    }

    async get(endpoint, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const url = queryString ? `${endpoint}?${queryString}` : endpoint;
        return this.request(url);
    }

    async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    async put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    async delete(endpoint) {
        return this.request(endpoint, {
            method: 'DELETE'
        });
    }

    clearCache() {
        this.cache.clear();
    }
}

// Form validation utility
const FormValidator = {
    patterns: {
        email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
        phone: /^\+?[\d\s-()]{10,}$/,
        url: /^https?:\/\/[\w\-]+(\.[\w\-]+)+[/#?]?.*$/,
        password: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/
    },

    validateEmail(email) {
        if (!email || typeof email !== 'string') return false;
        return this.patterns.email.test(email.trim());
    },

    validatePhone(phone) {
        if (!phone || typeof phone !== 'string') return false;
        const cleaned = phone.replace(/[\s-()]/g, '');
        return cleaned.length >= 10 && /^\+?\d+$/.test(cleaned);
    },

    validateURL(url) {
        if (!url || typeof url !== 'string') return false;
        return this.patterns.url.test(url.trim());
    },

    validatePassword(password) {
        if (!password || typeof password !== 'string') return false;
        return this.patterns.password.test(password);
    },

    validateRequired(value) {
        if (value === null || value === undefined) return false;
        if (typeof value === 'string') return value.trim().length > 0;
        if (Array.isArray(value)) return value.length > 0;
        return true;
    },

    validateLength(value, min, max) {
        if (!this.validateRequired(value)) return false;
        if (typeof value === 'string') {
            const length = value.trim().length;
            return length >= min && (max === undefined || length <= max);
        }
        if (Array.isArray(value)) {
            return value.length >= min && (max === undefined || value.length <= max);
        }
        return false;
    },

    validateForm(formData, rules) {
        const errors = {};

        for (const [field, fieldRules] of Object.entries(rules)) {
            const value = formData[field];

            for (const rule of fieldRules) {
                let isValid = true;
                let message = '';

                switch (rule.type) {
                    case 'required':
                        isValid = this.validateRequired(value);
                        message = `${field} is required`;
                        break;
                    case 'email':
                        isValid = this.validateEmail(value);
                        message = `${field} must be a valid email`;
                        break;
                    case 'phone':
                        isValid = this.validatePhone(value);
                        message = `${field} must be a valid phone number`;
                        break;
                    case 'url':
                        isValid = this.validateURL(value);
                        message = `${field} must be a valid URL`;
                        break;
                    case 'password':
                        isValid = this.validatePassword(value);
                        message = `${field} must contain at least 8 characters with uppercase, lowercase, number and special character`;
                        break;
                    case 'length':
                        isValid = this.validateLength(value, rule.min, rule.max);
                        message = `${field} must be between ${rule.min} and ${rule.max} characters`;
                        break;
                    case 'custom':
                        isValid = rule.validator(value, formData);
                        message = rule.message || `${field} is invalid`;
                        break;
                }

                if (!isValid) {
                    errors[field] = message;
                    break;
                }
            }
        }

        return {
            isValid: Object.keys(errors).length === 0,
            errors
        };
    }
};

// Data transformation and processing
class DataProcessor {
    static groupBy(array, key) {
        return array.reduce((groups, item) => {
            const groupKey = item[key];
            if (!groups[groupKey]) {
                groups[groupKey] = [];
            }
            groups[groupKey].push(item);
            return groups;
        }, {});
    }

    static sortBy(array, key, order = 'asc') {
        return [...array].sort((a, b) => {
            const aVal = a[key];
            const bVal = b[key];

            if (typeof aVal === 'string' && typeof bVal === 'string') {
                return order === 'asc'
                    ? aVal.localeCompare(bVal)
                    : bVal.localeCompare(aVal);
            }

            return order === 'asc' ? aVal - bVal : bVal - aVal;
        });
    }

    static filterBy(array, filters) {
        return array.filter(item => {
            return Object.entries(filters).every(([key, value]) => {
                if (typeof value === 'function') {
                    return value(item[key]);
                }
                return item[key] === value;
            });
        });
    }

    static paginate(array, page = 1, pageSize = 10) {
        const startIndex = (page - 1) * pageSize;
        const endIndex = startIndex + pageSize;
        const totalPages = Math.ceil(array.length / pageSize);

        return {
            data: array.slice(startIndex, endIndex),
            pagination: {
                page,
                pageSize,
                totalItems: array.length,
                totalPages,
                hasNext: page < totalPages,
                hasPrev: page > 1
            }
        };
    }

    static deduplicate(array, key) {
        const seen = new Set();
        return array.filter(item => {
            const itemKey = key ? item[key] : JSON.stringify(item);
            if (seen.has(itemKey)) {
                return false;
            }
            seen.add(itemKey);
            return true;
        });
    }

    static flatten(array) {
        return array.reduce((flat, item) => {
            return flat.concat(Array.isArray(item) ? this.flatten(item) : item);
        }, []);
    }

    static chunk(array, size) {
        const chunks = [];
        for (let i = 0; i < array.length; i += size) {
            chunks.push(array.slice(i, i + size));
        }
        return chunks;
    }
}

// Event emitter implementation
class EventEmitter {
    constructor() {
        this.events = new Map();
    }

    on(event, listener) {
        if (!this.events.has(event)) {
            this.events.set(event, new Set());
        }
        this.events.get(event).add(listener);
        return this;
    }

    off(event, listener) {
        if (this.events.has(event)) {
            this.events.get(event).delete(listener);
            if (this.events.get(event).size === 0) {
                this.events.delete(event);
            }
        }
        return this;
    }

    emit(event, ...args) {
        if (this.events.has(event)) {
            this.events.get(event).forEach(listener => {
                try {
                    listener(...args);
                } catch (error) {
                    console.error(`Error in event listener for ${event}:`, error);
                }
            });
        }
        return this;
    }

    once(event, listener) {
        const onceWrapper = (...args) => {
            this.off(event, onceWrapper);
            listener(...args);
        };
        return this.on(event, onceWrapper);
    }

    listenerCount(event) {
        return this.events.has(event) ? this.events.get(event).size : 0;
    }

    eventNames() {
        return Array.from(this.events.keys());
    }
}

// Cache manager with TTL support
class CacheManager {
    constructor(defaultTTL = 300000) { // 5 minutes default
        this.cache = new Map();
        this.defaultTTL = defaultTTL;
    }

    set(key, value, ttl = this.defaultTTL) {
        const expiresAt = Date.now() + ttl;
        this.cache.set(key, {
            value,
            expiresAt
        });

        // Schedule cleanup if this is the first entry
        if (this.cache.size === 1) {
            this.scheduleCleanup();
        }
    }

    get(key) {
        const entry = this.cache.get(key);

        if (!entry) {
            return undefined;
        }

        if (Date.now() > entry.expiresAt) {
            this.cache.delete(key);
            return undefined;
        }

        return entry.value;
    }

    delete(key) {
        return this.cache.delete(key);
    }

    clear() {
        this.cache.clear();
    }

    has(key) {
        const entry = this.cache.get(key);
        if (entry && Date.now() > entry.expiresAt) {
            this.cache.delete(key);
            return false;
        }
        return this.cache.has(key);
    }

    size() {
        this.cleanup(); // Clean before reporting size
        return this.cache.size;
    }

    keys() {
        this.cleanup();
        return Array.from(this.cache.keys());
    }

    cleanup() {
        const now = Date.now();
        for (const [key, entry] of this.cache.entries()) {
            if (now > entry.expiresAt) {
                this.cache.delete(key);
            }
        }
    }

    scheduleCleanup() {
        setTimeout(() => {
            this.cleanup();
            if (this.cache.size > 0) {
                this.scheduleCleanup();
            }
        }, 60000); // Clean every minute
    }
}

// Utility functions for common operations
const Utils = {
    debounce(func, wait, immediate = false) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                timeout = null;
                if (!immediate) func(...args);
            };
            const callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            if (callNow) func(...args);
        };
    },

    throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    deepClone(obj) {
        if (obj === null || typeof obj !== 'object') return obj;
        if (obj instanceof Date) return new Date(obj.getTime());
        if (obj instanceof Array) return obj.map(item => this.deepClone(item));
        if (obj instanceof Object) {
            const clonedObj = {};
            for (const key in obj) {
                if (obj.hasOwnProperty(key)) {
                    clonedObj[key] = this.deepClone(obj[key]);
                }
            }
            return clonedObj;
        }
    },

    formatCurrency(amount, currency = 'USD', locale = 'en-US') {
        return new Intl.NumberFormat(locale, {
            style: 'currency',
            currency: currency
        }).format(amount);
    },

    formatDate(date, locale = 'en-US', options = {}) {
        const defaultOptions = {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        };
        return new Intl.DateTimeFormat(locale, { ...defaultOptions, ...options }).format(date);
    },

    generateId(length = 8) {
        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        let result = '';
        for (let i = 0; i < length; i++) {
            result += chars.charAt(Math.floor(Math.random() * chars.length));
        }
        return result;
    },

    isEmpty(value) {
        if (value === null || value === undefined) return true;
        if (typeof value === 'string') return value.trim().length === 0;
        if (Array.isArray(value)) return value.length === 0;
        if (typeof value === 'object') return Object.keys(value).length === 0;
        return false;
    },

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
};

// Example usage and testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        UserManager,
        ApiService,
        FormValidator,
        DataProcessor,
        EventEmitter,
        CacheManager,
        Utils
    };
}

// Demo usage
console.log('Real-world JavaScript patterns loaded successfully');