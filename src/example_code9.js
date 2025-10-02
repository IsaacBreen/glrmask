aaaaaaaa
// [x]: 1
// [Symbol('description')]: 2

let aaaaaaaa = 11111111111111111111111111111111;
const myObject = {
    // Numeric keys
    123: "numeric key", // Numeric keys
    // Computed property name
    [x]: "symbol key value",
    [Symbol('description')]: "symbol key value",
};
let aaaaaaaa = 11111111111111111111111111111111;
let aaaaaaaa = 11111111111111111111111111111111;
aaaaa;
//=========
// <></>

// =================================================================
//  JavaScript Parser Test Suite - "The Kitchen Sink"
//  This file contains a wide variety of JS syntax constructs
//  to test the robustness and completeness of a parser.
// =================================================================


// --- 1. Basic Declarations and Primitives ---

// Variable declarations
var a = 1;
let b = "hello";
const c = true;

// Reassignment
a = 2;
b = 'world';
// const c = false; // This should be a syntax error if your parser handles const correctly

// Primitive types
let myNumber = 123.45;
let myBigInt = 9007199254740991n;
let myStringSingle = 'This is a string.';
let myStringDouble = "This is also a string.";
let myBoolean = false;
let myNull = null;
let myUndefined = undefined;
let mySymbol = Symbol('description');

// Scientific and other numeric notations
let scientific = 5e3;
let scientificNegative = 1.23e-2;
let hex = 0xff;
let octal = 0o77;
let binary = 0b11;


// --- 2. Objects and Arrays ---

// Array literal with various types and a trailing comma
const myArray = [
    1,
    "text",
    null,
    { key: 'value' },
    [1, 2, 3],
];

// Object literal
const myObject = {
    stringKey: "value",
    "key-with-hyphen": 123,
    unquotedKey: true,
    123: "numeric key", // Numeric keys
    // Computed property name
    [mySymbol]: "symbol key value",
    // Trailing comma
};

// Array destructuring
let [x, y, ...rest] = myArray;
let [ , , z] = myArray; // Skipping elements

// Object destructuring
let { stringKey, unquotedKey: newName, nonExistent = 'default' } = myObject;


// --- 3. Operators ---

let num = 10;

// Arithmetic
let add = num + 5;
let sub = num - 5;
let mul = num * 5;
let div = num / 5;
let mod = num % 3;
let exp = num ** 2;

// Unary, Increment/Decrement
let preIncrement = ++num;
let postIncrement = num++;
let preDecrement = --num;
let postDecrement = num--;
let unaryPlus = + "10";
let unaryNegation = -num;

// Comparison
let eq = (5 == '5');
let strictEq = (5 === '5');
let notEq = (5 != 5);
let strictNotEq = (5 !== '5');
let gt = 10 > 5;
let lt = 10 < 5;
let gte = 10 >= 10;
let lte = 10 <= 5;

// Logical
let and = true && false;
let or = true || false;
let not = !true;

// Bitwise
let bitAnd = 5 & 1;
let bitOr = 5 | 1;
let bitXor = 5 ^ 1;
let bitNot = ~5;