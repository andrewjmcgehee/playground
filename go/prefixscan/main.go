package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

const N = 1048576
const RES_3 = 20
const RES_5 = 23
const RES_6 = 28
const RES_7 = 35
const RES_33 = 154
const RES_65536 = 295877
const RES_1048576 = 4717416

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

func nextPowerOfTwo(x int) int {
	if x > 0 && (x&(x-1)) == 0 {
		return x
	}
	result := 1
	for result < x {
		result <<= 1
	}
	return result
}

func log2(x int) int {
	result := 0
	for x > 1 {
		x >>= 1
		result++
	}
	return result
}

func add(i, j int, res []int, done chan bool) {
	res[i] += res[j]
	done <- true
}

func constrainSize(arr []int) []int {
	n := nextPowerOfTwo(len(arr))
	out := make([]int, n)
	copy(out, arr)
	return out
}

func upSweep(arr []int) {
	n := len(arr)
	done := make(chan bool)
	for d := 0; d < log2(n); d++ {
		workers := 0
		for k := 0; k < n; k += 1 << (d + 1) {
			workers++
			go add(k+(1<<(d+1))-1, k+(1<<d)-1, arr, done)
		}
		for range workers {
			<-done
		}
	}
}

func downSweep(arr []int) {
	n := len(arr)
	sum := arr[n-1]
	arr[n-1] = 0
	done := make(chan bool)
	for d := log2(n) - 1; d >= 0; d-- {
		workers := 0
		for k := 0; k < n; k += 1 << (d + 1) {
			workers++
			go func(k int) {
				t := arr[k+(1<<d)-1]
				arr[k+(1<<d)-1] = arr[k+(1<<(d+1))-1]
				arr[k+(1<<(d+1))-1] += t
				done <- true
			}(k)
		}
		for range workers {
			<-done
		}
	}
	copy(arr[1:], arr)
	arr[n-1] = sum
}

func prefixScanLogParallel(src []int) {
	arr := constrainSize(src)
	upSweep(arr)
	downSweep(arr)
	copy(src, arr)
}

func prefixScanChunkedParallel(src []int) {
	p := min(32, len(src)/2)
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
	if res[len(res)-1] != RES_1048576 {
		return fmt.Errorf("expected %d got %d", RES_1048576, res[len(res)-1])
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
	prefixScanChunkedParallel(src)
	dur = time.Since(start)
	err = checkResult(src)
	if err != nil {
		panic(err)
	}
	fmt.Printf("reduce parallel: %v\n", dur)
}
