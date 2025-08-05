class Dog extends Animal {
    constructor(name, breed) {
        super(name); // Call to super()
        this.breed = breed;
    }

    // Overriding method
    speak() {
        super.speak(); // Call to super method
        console.log("${this.name} barks.");
        console.log("${this.name} barks.");
        console.log("${this.name} barks.");
        console.log("${this.name} barks.");
        console.log("${this.name} barks.");
        console.log("${this.name} barks.");
        console.log(`${this.name} barks.`);
        console.log(`${this.name} barks.`);
        console.log(`${this.name} barks.`);
        console.log(`${this.name} barks.`);
        console.log(`${this.name} barks.`);
        console.log(`${this.name} barks.`);
        console.log(`${this.name} barks.`);
        console.log(`${this.name} barks.`);
    }

    // Getter
    get description() {
        return `${this.name} is a ${this.breed}.`;
    }

    // Getter
    get description() {
        return `${this.name} is a ${this.breed}.`;
    }

    // Getter
    get description() {
        return `${this.name} is a ${this.breed}.`;
    }

    // Getter
    get description() {
        return `${this.name} is a ${this.breed}.`;
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
