// Package compute is a dependency-free, cgo-free standalone port of the
// PowerEngine cooperative compute engine. It provides the float32 primitives the
// native diffusion and transformer engines run on — SAXPY, elementwise add/scale,
// biased matrix multiply, and fused scaled-dot-product attention — fanned out
// across CPU workers. The upstream engine also dispatches to Metal GPUs; this
// port keeps the identical API and CPU kernels and stubs the GPU backend so it
// builds anywhere with no cgo (see gpu_stub.go).
package compute

import (
	"errors"
	"fmt"
)

var (
	ErrShapeMismatch = errors.New("compute: shape mismatch")
	ErrEmptyDevice   = errors.New("compute: no execution device available")
)

func checkSameLength(op string, n int, vectors ...namedLen) error {
	for _, vector := range vectors {
		if vector.n != n {
			return fmt.Errorf("%w: %s requires len(%s)=%d, got %d", ErrShapeMismatch, op, vector.name, n, vector.n)
		}
	}
	return nil
}

type namedLen struct {
	name string
	n    int
}
