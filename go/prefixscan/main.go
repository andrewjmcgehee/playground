package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

const N = 65536
const RES_33 = 154
const RES_65536 = 295877

func getPreArray() []int {
	r := rand.New(rand.NewSource(42))
	arr := make([]int, N)
	for i := 0; i < N; i++ {
		arr[i] = r.Intn(10)
	}
	return arr
}

func prefixScanLinear(src []int) {
	for i := 1; i < len(src); i++ {
		src[i] = src[i-1] + src[i]
	}
}

func getChunkBounds(p, length int) [][]int {
	bounds := [][]int{}
	chunkSize := length / p
	remainder := length % p
	for i := 0; i < p; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if i == p-1 {
			end += remainder
		}
		if end > length {
			end = length
		}
		bounds = append(bounds, []int{start, end})
	}
	return bounds
}

func prefixScanLogParallel(res []int) {
	wg := sync.WaitGroup{}
	stride := 2
	for stride <= len(res) {
		for i := stride - 1; i < len(res); i += stride {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				res[i] = res[i] + res[i-stride/2]
			}(i)
		}
		wg.Wait()
		stride *= 2
	}
	if len(res)%2 != 0 {
		res[len(res)-1] = res[len(res)-1] + res[len(res)-2]
	}
}

func prefixScanReduceParallel(src []int) {
	p := 32
	bounds := getChunkBounds(p, len(src))
	wg := sync.WaitGroup{}
	mem := make([]int, p)
	for i := 0; i < p; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			start, end := bounds[i][0], bounds[i][1]
			prefixScanLinear(src[start:end])
			mem[i] = src[end-1]
		}(i)
	}
	wg.Wait()
	prefixScanLinear(mem)
	src[len(src)-1] = mem[p-1]
}

func checkResult(res []int) error {
	if res[len(res)-1] != RES_65536 {
		return fmt.Errorf("expected %d got %d", RES_65536, res[len(res)-1])
	}
	return nil
}

func main() {
	src := getPreArray()
	start := time.Now()
	prefixScanLinear(src)
	dur := time.Since(start)
	err := checkResult(src)
	if err != nil {
		panic(err)
	}
	fmt.Printf("linear: %v\n", dur)
	src = getPreArray()
	start = time.Now()
	prefixScanLogParallel(src)
	dur = time.Since(start)
	err = checkResult(src)
	if err != nil {
		panic(err)
	}
	fmt.Printf("log n parallel: %v\n", dur)
	src = getPreArray()
	start = time.Now()
	prefixScanReduceParallel(src)
	dur = time.Since(start)
	err = checkResult(src)
	if err != nil {
		panic(err)
	}
	fmt.Printf("reduce parallel: %v\n", dur)
}
