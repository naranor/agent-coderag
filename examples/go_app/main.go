package main
import "fmt"
import "net/http"

type Server struct{}

// Start begins the HTTP server on the given port.
// This is a named method.
func (s *Server) Start(port int) {
    fmt.Printf("Server starting on %d\n", port)

    // Anonymous function assigned to a variable
    handler := func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprint(w, "Hello!")
    }

    http.HandleFunc("/", handler)
    http.ListenAndServe(fmt.Sprintf(":%d", port), nil)
}
