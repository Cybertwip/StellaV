package compute

// This standalone build ships no native GPU backend, so it has no cgo and builds
// on any platform. The upstream engine plugs a Metal backend in here; the API and
// the cooperative CPU/GPU plan machinery are unchanged, so a GPU backend can be
// reintroduced without touching callers.

func newMetalBackends(Options) ([]gpuBackend, error) { return nil, nil }

func listGPUDevices() []Device { return nil }
