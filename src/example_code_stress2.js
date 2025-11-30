// JavaScript Stress Test - 500+ lines of sequential statements
// This file contains various JavaScript patterns and operations for stress testing

// Basic variable declarations and assignments
let counter = 0;
const maxCount = 1000;
let totalSum = 0;
let product = 1;
let stringResult = "";
let booleanFlags = [true, false, true, false, true];
let numberArray = [];
let objectCollection = {};
let functionResults = [];

// Arithmetic operations
counter = counter + 1;
totalSum = totalSum + counter;
product = product * 2;
let divisionResult = 100 / counter;
let modulusResult = counter % 7;
let exponentResult = Math.pow(2, counter);
let sqrtResult = Math.sqrt(counter);
let absResult = Math.abs(-counter);
let floorResult = Math.floor(divisionResult);
let ceilResult = Math.ceil(divisionResult);
let roundResult = Math.round(divisionResult);

// String operations
stringResult = stringResult + "iteration_" + counter;
stringResult = stringResult.toUpperCase();
stringResult = stringResult.toLowerCase();
let stringLength = stringResult.length;
let substringResult = stringResult.substring(0, 5);
let charAtResult = stringResult.charAt(3);
let indexOfResult = stringResult.indexOf("iteration");
let lastIndexOfResult = stringResult.lastIndexOf("iteration");
let replaceResult = stringResult.replace("iteration", "cycle");
let splitResult = stringResult.split("_");
let trimResult = stringResult.trim();
let startsWithResult = stringResult.startsWith("iteration");
let endsWithResult = stringResult.endsWith("counter");
let includesResult = stringResult.includes("cycle");

// Array operations
numberArray.push(counter);
numberArray.unshift(counter * 2);
let poppedValue = numberArray.pop();
let shiftedValue = numberArray.shift();
numberArray.splice(2, 0, counter * 3);
let slicedArray = numberArray.slice(1, 4);
let concatenatedArray = numberArray.concat([100, 200, 300]);
let joinedString = numberArray.join(",");
let reversedArray = numberArray.reverse();
let sortedArray = numberArray.sort((a, b) => a - b);
let filteredArray = numberArray.filter(x => x > 50);
let mappedArray = numberArray.map(x => x * 2);
let reducedValue = numberArray.reduce((acc, val) => acc + val, 0);
let foundValue = numberArray.find(x => x > 100);
let foundIndex = numberArray.findIndex(x => x > 100);
let everyResult = numberArray.every(x => x > 0);
let someResult = numberArray.some(x => x > 1000);

// Object operations
objectCollection["key_" + counter] = counter;
objectCollection["string_" + counter] = stringResult;
objectCollection["array_" + counter] = numberArray.slice();
let objectKeys = Object.keys(objectCollection);
let objectValues = Object.values(objectCollection);
let objectEntries = Object.entries(objectCollection);
let hasProperty = objectCollection.hasOwnProperty("key_1");
let propertyCount = Object.keys(objectCollection).length;

// Date operations
let currentDate = new Date();
let timestamp = currentDate.getTime();
let year = currentDate.getFullYear();
let month = currentDate.getMonth();
let day = currentDate.getDate();
let hours = currentDate.getHours();
let minutes = currentDate.getMinutes();
let seconds = currentDate.getSeconds();
let milliseconds = currentDate.getMilliseconds();
let dayOfWeek = currentDate.getDay();
let isoString = currentDate.toISOString();
let localeString = currentDate.toLocaleString();
let utcHours = currentDate.getUTCHours();
let timezoneOffset = currentDate.getTimezoneOffset();

// Math operations
let randomValue = Math.random();
let sinValue = Math.sin(counter);
let cosValue = Math.cos(counter);
let tanValue = Math.tan(counter);
let logValue = Math.log(counter + 1);
let log10Value = Math.log10(counter + 1);
let expValue = Math.exp(counter / 10);
let maxValue = Math.max(...numberArray);
let minValue = Math.min(...numberArray);
let piValue = Math.PI;
let eValue = Math.E;

// Conditional operations
if (counter > 50) {
    totalSum = totalSum * 1.1;
} else if (counter > 25) {
    totalSum = totalSum * 1.05;
} else {
    totalSum = totalSum * 1.02;
}

if (stringResult.length > 100) {
    stringResult = stringResult.substring(0, 100);
}

if (numberArray.length > 50) {
    numberArray = numberArray.slice(0, 50);
}

if (Object.keys(objectCollection).length > 100) {
    let keys = Object.keys(objectCollection);
    for (let i = 0; i < 50; i++) {
        delete objectCollection[keys[i]];
    }
}

// Loop operations
for (let i = 0; i < 10; i++) {
    let tempValue = i * counter;
    totalSum += tempValue;
    product *= (tempValue + 1);
    stringResult += "_loop" + i;
}

for (let key in objectCollection) {
    if (objectCollection.hasOwnProperty(key)) {
        let value = objectCollection[key];
        if (typeof value === 'number') {
            objectCollection[key] = value * 1.1;
        }
    }
}

for (let item of numberArray) {
    if (item % 2 === 0) {
        totalSum += item;
    } else {
        product *= item;
    }
}

let whileCounter = 0;
while (whileCounter < 5) {
    let tempCalc = Math.pow(whileCounter, 2);
    totalSum += tempCalc;
    whileCounter++;
}

let doCounter = 0;
do {
    let tempCalc = Math.sqrt(doCounter);
    product *= (tempCalc + 1);
    doCounter++;
} while (doCounter < 5);

// Function definitions and calls
function calculateFactorial(n) {
    if (n <= 1) return 1;
    return n * calculateFactorial(n - 1);
}

function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

function isPrime(num) {
    if (num <= 1) return false;
    if (num <= 3) return true;
    if (num % 2 === 0 || num % 3 === 0) return false;
    for (let i = 5; i * i <= num; i += 6) {
        if (num % i === 0 || num % (i + 2) === 0) return false;
    }
    return true;
}

function processArray(arr) {
    return arr
        .filter(x => x > 0)
        .map(x => x * 2)
        .reduce((acc, val) => acc + val, 0);
}

function createObject(name, value, type) {
    return {
        name: name,
        value: value,
        type: type,
        timestamp: Date.now(),
        processed: false
    };
}

// Function calls
let factorialResult = calculateFactorial(10);
let fibonacciResult = fibonacci(15);
let primeCheck = isPrime(997);
let arrayProcessResult = processArray(numberArray);
let newObject = createObject("test_object", counter, "number");

// More complex operations
function complexCalculation(a, b, c) {
    let result = 0;
    for (let i = 0; i < a; i++) {
        result += Math.sin(i * b) * Math.cos(i * c);
    }
    return result;
}

function stringManipulation(str) {
    let words = str.split(/\s+/);
    let processed = words
        .filter(word => word.length > 3)
        .map(word => word.toUpperCase())
        .reverse()
        .join(" ");
    return processed;
}

function arrayTransformations(arr) {
    return arr
        .filter(x => x % 2 === 0)
        .map(x => ({ value: x, squared: x * x, cubed: x * x * x }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 10);
}

// Execute complex functions
let complexResult = complexCalculation(20, Math.PI / 10, Math.PI / 20);
let manipulatedString = stringManipulation(stringResult);
let transformedArray = arrayTransformations(numberArray);

// Error handling operations
try {
    let riskyDivision = 100 / (counter - 50);
    functionResults.push(riskyDivision);
} catch (error) {
    functionResults.push(0);
}

try {
    let undefinedAccess = objectCollection.nonExistentProperty.someMethod();
} catch (error) {
    functionResults.push("caught_error");
}

try {
    let jsonParse = JSON.parse('{"valid": "json"}');
    functionResults.push(jsonParse.valid);
} catch (error) {
    functionResults.push("parse_error");
}

// Regular expression operations
let regexPattern = /iteration_\d+/g;
let regexMatches = stringResult.match(regexPattern);
let regexTest = regexPattern.test(stringResult);
let regexReplace = stringResult.replace(/iteration/g, "process");
let regexSplit = stringResult.split(/[_\d]+/);

let emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
let emailTest = emailRegex.test("test@example.com");

let numberRegex = /\d+/g;
let numberMatches = stringResult.match(numberRegex);

// JSON operations
let jsonString = JSON.stringify(objectCollection);
let parsedObject = JSON.parse(jsonString);
let jsonSize = jsonString.length;

let complexObject = {
    metadata: {
        created: new Date().toISOString(),
        version: "1.0.0",
        author: "stress_test"
    },
    data: {
        numbers: numberArray,
        strings: [stringResult, manipulatedString],
        objects: transformedArray,
        calculations: {
            totalSum: totalSum,
            product: product,
            factorial: factorialResult,
            fibonacci: fibonacciResult
        }
    },
    flags: booleanFlags
};

let complexJson = JSON.stringify(complexObject, null, 2);

// Set and Map operations
let numberSet = new Set(numberArray);
numberSet.add(counter * 10);
numberSet.add(counter * 20);
let setSize = numberSet.size;
let setHas = numberSet.has(counter);
numberSet.delete(counter);

let stringMap = new Map();
stringMap.set("counter", counter);
stringMap.set("totalSum", totalSum);
stringMap.set("product", product);
stringMap.set("stringResult", stringResult);
let mapSize = stringMap.size;
let mapGet = stringMap.get("counter");
let mapHas = stringMap.has("totalSum");
stringMap.delete("product");

// Promise and async operations (simulated)
function simulateAsyncOperation(value) {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve(value * 2);
        }, 0);
    });
}

async function executeAsyncOperations() {
    let results = [];
    for (let i = 0; i < 5; i++) {
        let result = await simulateAsyncOperation(i + counter);
        results.push(result);
    }
    return results;
}

// Execute async operations (will complete immediately due to 0 timeout)
let asyncResults = [];
(async () => {
    asyncResults = await executeAsyncOperations();
})();

// More mathematical operations
let hyperbolicSin = Math.sinh(counter / 10);
let hyperbolicCos = Math.cosh(counter / 10);
let hyperbolicTan = Math.tanh(counter / 10);
let arcSin = Math.asin(counter / 100);
let arcCos = Math.acos(counter / 100);
let arcTan = Math.atan(counter / 100);
let arcTan2 = Math.atan2(counter, counter + 1);

// Bitwise operations
let bitwiseAnd = counter & 15;
let bitwiseOr = counter | 8;
let bitwiseXor = counter ^ 12;
let bitwiseNot = ~counter;
let leftShift = counter << 2;
let rightShift = counter >> 1;
let zeroFillRightShift = counter >>> 1;

// Type conversion operations
let stringToNumber = parseInt("123" + counter);
let stringToFloat = parseFloat("123." + counter);
let numberToString = counter.toString();
let numberToHex = counter.toString(16);
let numberToBinary = counter.toString(2);
let booleanToString = true.toString();
let arrayToString = numberArray.toString();
let objectToString = objectCollection.toString();

// More array methods
let arrayEvery = numberArray.every(x => typeof x === 'number');
let arraySome = numberArray.some(x => x > 1000);
let arrayFind = numberArray.find(x => x > 500);
let arrayFindIndex = numberArray.findIndex(x => x > 500);
let arrayIncludes = numberArray.includes(counter);
let arrayIndexOf = numberArray.indexOf(counter);
let arrayLastIndexOf = numberArray.lastIndexOf(counter);

// String padding and trimming
let paddedStart = stringResult.padStart(50, "*");
let paddedEnd = stringResult.padEnd(50, "*");
let trimmedStart = stringResult.trimStart();
let trimmedEnd = stringResult.trimEnd();

// More object operations
let objectAssign = Object.assign({}, objectCollection, { newProperty: "added" });
let objectFreeze = Object.freeze({ constant: "value" });
let objectSeal = Object.seal({ modifiable: "value" });

// Date formatting and manipulation
let formattedDate = currentDate.toLocaleDateString('en-US');
let formattedTime = currentDate.toLocaleTimeString('en-US');
let utcDate = currentDate.toUTCString();
let dateString = currentDate.toString();

// Create future and past dates
let futureDate = new Date(currentDate.getTime() + 86400000); // +1 day
let pastDate = new Date(currentDate.getTime() - 86400000); // -1 day
let dateDifference = futureDate.getTime() - pastDate.getTime();

// More mathematical constants and functions
let ln2 = Math.LN2;
let ln10 = Math.LN10;
let log2e = Math.LOG2E;
let log10e = Math.LOG10E;
let sqrt1_2 = Math.SQRT1_2;
let sqrt2 = Math.SQRT2;

// Random number generation with different ranges
let randomInt = Math.floor(Math.random() * 100) + 1;
let randomFloat = Math.random() * 1000;
let randomInRange = Math.random() * (100 - 50) + 50;

// Trigonometric functions with different angles
let degreesToRadians = counter * Math.PI / 180;
let sinDegrees = Math.sin(degreesToRadians);
let cosDegrees = Math.cos(degreesToRadians);
let tanDegrees = Math.tan(degreesToRadians);

// Logarithmic functions with different bases
let naturalLog = Math.log(counter + 1);
let base10Log = Math.log10(counter + 1);
let base2Log = Math.log2(counter + 1);

// Exponential functions
let exponential = Math.exp(counter / 10);
let exponentialMinusOne = Math.expm1(counter / 10);

// More string operations
let charCode = stringResult.charCodeAt(0);
let fromCharCode = String.fromCharCode(65 + counter % 26);
let codePointAt = stringResult.codePointAt(0);
let normalized = stringResult.normalize();

// Array buffer and typed arrays
let buffer = new ArrayBuffer(16);
let int32View = new Int32Array(buffer);
let float64View = new Float64Array(buffer);

// Fill typed arrays
for (let i = 0; i < int32View.length; i++) {
    int32View[i] = i * counter;
}

for (let i = 0; i < float64View.length; i++) {
    float64View[i] = Math.random() * 100;
}

// More set operations
let setUnion = new Set([...numberSet, ...new Set([100, 200, 300])]);
let setIntersection = new Set([...numberSet].filter(x => new Set([100, 200, 300]).has(x)));
let setDifference = new Set([...numberSet].filter(x => !new Set([100, 200, 300]).has(x)));

// More map operations
let mapEntries = stringMap.entries();
let mapKeys = stringMap.keys();
let mapValues = stringMap.values();

// Final summary calculations
let finalAverage = totalSum / (counter + 1);
let finalVariance = numberArray.reduce((acc, val) => acc + Math.pow(val - finalAverage, 2), 0) / numberArray.length;
let finalStdDev = Math.sqrt(finalVariance);

// Create final result object
let finalResult = {
    summary: {
        totalIterations: counter,
        finalSum: totalSum,
        finalProduct: product,
        average: finalAverage,
        standardDeviation: finalStdDev,
        timestamp: new Date().toISOString()
    },
    data: {
        numbers: numberArray,
        strings: [stringResult, manipulatedString],
        objects: objectCollection,
        arrays: [transformedArray, slicedArray, filteredArray]
    },
    metadata: {
        executionTime: Date.now() - timestamp,
        memoryUsage: process.memoryUsage ? process.memoryUsage() : {},
        platform: typeof process !== 'undefined' ? process.platform : 'browser'
    }
};

// Export final result if in module context
if (typeof module !== 'undefined' && module.exports) {
    module.exports = finalResult;
}

// Final console output for verification
console.log("Stress test completed successfully");
console.log(`Total iterations: ${counter}`);
console.log(`Final sum: ${totalSum}`);
console.log(`Final product: ${product}`);
console.log(`String result length: ${stringResult.length}`);
console.log(`Array length: ${numberArray.length}`);
console.log(`Object properties: ${Object.keys(objectCollection).length}`);