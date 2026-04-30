# Python Code Quality Reference

## SOLID Principles

### S: Single Responsibility Principle
A class or function should have only one reason to change. Each component should do one thing and do it well.

### O: Open/Closed Principle
Software entities should be open for extension but closed for modification. Use inheritance or composition to add new functionality without changing existing code.

### L: Liskov Substitution Principle
Subclasses should be substitutable for their base classes. Derived classes must be able to replace their parent classes without affecting the correctness of the program.

### I: Interface Segregation Principle
Clients should not be forced to depend on methods they do not use. Prefer many small, specific interfaces over one large, general-purpose one.

### D: Dependency Inversion Principle
Depend on abstractions, not on concretions. High-level modules should not depend on low-level modules; both should depend on abstractions.

## Best Practices

### PEP 8 Compliance
- Indentation: Use 4 spaces per indentation level.
- Line Length: Limit all lines to a maximum of 79 characters (or 88/100/120 depending on team agreement).
- Naming: `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_CASE` for constants.

### Documentation
- Use docstrings for all modules, classes, and public methods.
- Follow the Google, NumPy, or Sphinx docstring format.

### Type Hinting
- Use Python 3.9+ type hints (`list[int]`, `dict[str, str]`).
- For older versions, use the `typing` module.

### Error Handling
- Use specific exceptions, never `except Exception: pass`.
- Use `try...except...finally` for resource cleanup.

## Common Anti-patterns
- **God Objects**: Classes that try to do everything.
- **Shotgun Surgery**: A single change requires modifying many small parts across the codebase.
- **Deep Nesting**: Avoid more than 3 levels of indentation.
- **Magic Strings/Numbers**: Use constants instead.
- **Premature Optimization**: Prioritize readability and correctness over performance unless measured.
