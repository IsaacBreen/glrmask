let aaaaaaaa = 11111111111111111111111111111111;
let aaaaaaaa = 11111111111111111111111111111111;
let aaaaaaaa = 11111111111111111111111111111111;
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
let leftShift = 5 << 1;
let rightShift = 5 >> 1;
let zeroFillRightShift = -5 >>> 1;

// Ternary
let ternary = (num > 10) ? "greater" : "less or equal";

// Other operators
let type = typeof num;
let isInstance = myArray instanceof Array;
delete myObject.stringKey;
let hasKey = "unquotedKey" in myObject;


// --- 4. Control Flow ---

// if / else if / else
if (a === 1) {
    console.log('one');
} else if (a === 2) {
    console.log('two');
} else {
    console.log('other');
}

// switch statement with fall-through
switch (a) {
    case 1:
        console.log('fall-through');
    case 2:
        console.log('is two');
        break;
    default:
        console.log('default case');
}

// for loop
for (let i = 0; i < 5; i++) {
    if (i === 3) continue; // test continue
    if (i === 4) break;    // test break
}

// while loop
let j = 0;
while (j < 3) {
    j++;
}

// do-while loop
let k = 0;
do {
    k++;
} while (k < 3);

// for...in (for object keys)
for (const key in myObject) {
    console.log(key);
}

// for...of (for iterable values)
for (const value of myArray) {
    console.log(value);
}

// try / catch / finally
try {
    throw new Error("Test error");
} catch (e) {
    console.error(e.message);
} finally {
    console.log("This always runs.");
}

// Labeled statements
outer_loop:
for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
        if (i === 1 && j === 1) {
            break outer_loop;
        }
    }
}


// --- 5. Functions and `this` ---

// Function declaration (hoisted)
function declaredFunction(p1, p2 = 'default') {
    return p1 + p2;
}

// Function expression (anonymous)
const exprFunction = function(a, b) {
    return a * b;
};

// Function expression (named)
const namedExprFunction = function innerName(a, b) {
    return a - b;
};

// Arrow functions
const arrow1 = () => "hello";
const arrow2 = x => x * x;
const arrow3 = (x, y) => {
    const sum = x + y;
    return sum;
};

// IIFE (Immediately Invoked Function Expression)
(function() {
    console.log('IIFE executed!');
})();

// `this` keyword context
const thisTest = {
    prop: 42,
    func: function() {
        // `this` refers to thisTest
        return this.prop;
    },
    arrowFunc: () => {
        // `this` is lexically scoped (will be window/global or undefined in strict mode)
        return this.prop;
    }
};

// Generator function
function* idMaker() {
    let index = 0;
    while (true) {
        yield index++;
    }
}
const gen = idMaker();
console.log(gen.next().value);

// Async/await
async function fetchData() {
    try {
        const response = await Promise.resolve('data');
        return response;
    } catch (err) {
        console.error(err);
    }
}


// --- 6. Classes and Prototypes ---

// ES6 Class
class Animal {
    #privateField = "secret"; // Private class field

    constructor(name) {
        this.name = name;
    }

    speak() {
        console.log(`${this.name} makes a noise. Secret is ${this.#privateField}`);
    }

    static info() {
        return "This is an Animal class.";
    }
}

class Dog extends Animal {
    constructor(name, breed) {
        super(name); // Call to super()
        this.breed = breed;
    }

    // Overriding method
    speak() {
        super.speak(); // Call to super method
        console.log(`${this.name} barks.`);
    }

    // Getter
    get description() {
        return `${this.name} is a ${this.breed}.`;
    }

    // Setter
    set nickname(nick) {
        this._nickname = nick;
    }
}

const d = new Dog('Milo', 'Golden Retriever');
d.speak();
console.log(Dog.info());


// --- 7. Modern ES6+ and Edge Cases ---

// Template literals
const personName = "Tester";
const template = `Hello, ${personName}! The result is ${10 + 5}.`;

// Tagged template literals
function myTag(strings, ...values) {
    return strings[0] + values[0] + strings[1];
}
const taggedResult = myTag`The value is ${100} dollars.`;

// Optional Chaining and Nullish Coalescing
const deepObject = { user: { address: { street: '123 Main St' } } };
const street = deepObject?.user?.address?.street;
const zip = deepObject?.user?.address?.zip ?? '00000';

// Regular Expressions
const regexLiteral = /ab+c/gi;
const regexConstructor = new RegExp('^\\d+$', 'u');
// A tricky case for a lexer: a division operator vs. a regex literal
let result = 10 / 2; // This is division
let regex = /foo/g;  // This is a regex

// Automatic Semicolon Insertion (ASI) - Tricky cases
let x_asi = 1
let y_asi = 2
// The above should parse as two separate statements.

let z_asi = x_asi
(y_asi).toString()
// The above should parse as `x_asi(y_asi).toString()`, a function call,
// which will cause a runtime error but is syntactically valid.

// `with` statement (deprecated, but must be parsed)
with (myObject) {
    console.log(unquotedKey);
}

// Comments in weird places
const weirdComment = [1, /* a comment */ 2, 3];
function weirdCommentFunc(/* arg1 */ p1, p2 /* arg2 */) {
    return p1; // single line
}

// --- 8. ES Modules (Special Case) ---
// NOTE: The following lines are syntactically valid ONLY in a module context.
// A robust parser should be able to handle this, often via a "sourceType: 'module'" flag.
// They are commented out here to allow this file to run as a standard script.

/*
import defaultExport from "module-name";
import * as name from "module-name";
import { export1, export2 as alias2 } from "module-name";
import("dynamic-module").then(mod => console.log(mod));

export const myExportedVar = 123;
export default function() {
    console.log('default export');
}
*/

// --- 9. Deeply Nested Statements ---
// To test parser recursion depth and handling of complex block structures.
function deeplyNestedTest(depth) {
    if (depth > 0) {
        for (let i = 0; i < 1; i++) {
            while (i < 1) {
                try {
                    switch (i) {
                        case 0:
                            if (true) {
                                let a = 1;
                                {
                                    {
                                        {
                                            deeplyNestedTest(depth - 1);
                                        }
                                    }
                                }
                            }
                            break;
                    }
                } catch (e) { /* ignore */ }
            }
        }
    }
}
