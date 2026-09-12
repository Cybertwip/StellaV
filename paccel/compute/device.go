package compute

import (
	"fmt"
	"runtime"
	"strings"
)

// DeviceKind identifies the execution class used by a work segment.
type DeviceKind string

const (
	DeviceCPU DeviceKind = "cpu"
	DeviceGPU DeviceKind = "gpu"
)

// Device describes an execution target known to the compute package.
type Device struct {
	Kind          DeviceKind
	ID            int
	Name          string
	Vendor        string
	Backend       string
	LowPower      bool
	Headless      bool
	Removable     bool
	UnifiedMemory bool
	RegistryID    uint64
}

func (d Device) Label() string {
	if d.Backend == "" {
		return d.Name
	}
	return fmt.Sprintf("%s/%s", d.Backend, d.Name)
}

func cpuDevice() Device {
	return Device{
		Kind:    DeviceCPU,
		ID:      0,
		Name:    fmt.Sprintf("Go CPU %s/%s", runtime.GOOS, runtime.GOARCH),
		Vendor:  "runtime",
		Backend: "go",
	}
}

// ListDevices returns the CPU backend and any native GPU backends available on
// the current host (none in this CPU-only standalone build).
func ListDevices() []Device {
	out := []Device{cpuDevice()}
	out = append(out, listGPUDevices()...)
	return out
}

func vendorFromName(name string) string {
	low := strings.ToLower(name)
	switch {
	case strings.Contains(low, "intel"):
		return "Intel"
	case strings.Contains(low, "amd"), strings.Contains(low, "radeon"):
		return "AMD"
	case strings.Contains(low, "apple"):
		return "Apple"
	case strings.Contains(low, "nvidia"):
		return "NVIDIA"
	default:
		return "unknown"
	}
}
