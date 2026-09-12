package compute

// Segment is a contiguous range assigned to one execution device.
type Segment struct {
	Device Device
	Start  int
	End    int
}

func (s Segment) Len() int {
	if s.End <= s.Start {
		return 0
	}
	return s.End - s.Start
}

// Plan describes how an operation is split across CPU and GPU devices. In this
// standalone (CPU-only) build every plan is a single CPU segment.
type Plan struct {
	Total    int
	Segments []Segment
}

func (p Plan) CPUItems() int { return p.itemsFor(DeviceCPU) }
func (p Plan) GPUItems() int { return p.itemsFor(DeviceGPU) }

func (p Plan) itemsFor(kind DeviceKind) int {
	total := 0
	for _, segment := range p.Segments {
		if segment.Device.Kind == kind {
			total += segment.Len()
		}
	}
	return total
}
