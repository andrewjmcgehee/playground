package main

import (
	"context"
	"fmt"
	"time"
)

func producerA(aChan chan string) {
	for range 5 {
		time.Sleep(time.Millisecond * 100)
		aChan <- "a"
	}
}

func producerB(bChan chan string) {
	for range 5 {
		time.Sleep(time.Millisecond * 200)
		bChan <- "b"
	}
}

func producerC(cChan chan string) {
	for range 5 {
		time.Sleep(time.Millisecond * 300)
		cChan <- "c"
	}
}

func main() {
	aChan := make(chan string)
	bChan := make(chan string)
	cChan := make(chan string)
	go producerA(aChan)
	go producerB(bChan)
	go producerC(cChan)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second*2)
	defer cancel()

	var data string
loop:
	for {
		select {
		case data = <-aChan:
			fmt.Println(data)
		case data = <-bChan:
			fmt.Println(data)
		case data = <-cChan:
			fmt.Println(data)
		case <-ctx.Done():
			break loop
		}
	}
}
