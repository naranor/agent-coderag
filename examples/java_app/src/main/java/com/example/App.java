package com.example;
import com.google.common.collect.Lists;
import java.util.List;

public class App {
    public void processData() {
        List<String> items = Lists.newArrayList("one", "two");
        System.out.println("Processing: " + items.size());
    }
}
