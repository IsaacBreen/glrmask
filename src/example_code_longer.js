// This is a longer JavaScript example to benchmark
function fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(x => x * 2);
console.log(doubled);

class Calculator {
    constructor(value = 0) {
        this.value = value;
    }
    
    add(n) {
        this.value += n;
        return this;
    }
    
    multiply(n) {
        this.value *= n;
        return this;
    }
}

const calc = new Calculator(5);
calc.add(3).multiply(2);
