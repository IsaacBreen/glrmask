const Base = class { base() { return 1; } };
const Derived = class Named extends Base {
    constructor() { super(); }
    method() { return super.base(); }
};
const functionExpression = async function* namedGenerator(value = 1) {
    yield await value;
};
const instance = new Derived();
const iterator = functionExpression(2);