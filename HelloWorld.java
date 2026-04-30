import java.util.List;
import java.util.ArrayList;

public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }

    public void greet(String name) {
        System.out.println("Hello, " + name);
    }
}

interface Greeter {
    void sayHello();
}
