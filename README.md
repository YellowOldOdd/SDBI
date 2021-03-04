# Simple Dynamic Batching Inference

## 解决了什么问题？
众所周知，Batch对于GPU上深度学习模型的运行效率影响很大。。。

是在Inference时。搜索、推荐等服务自带比较大的batch。问题不大。但更多场景面临的往往是稀碎的请求(比如一次一张图)。

如果想提高服务的吞吐，把稀碎的请求动态攒成Batch再送GPU处理就是刚需。

NV的Triton包含了Dynamic Batching功能。我也用cpp写过一版。但是发现在部署、特别是给别人用python来调用的时候，始终是比较麻烦的。比如要各种配置环境或用NGC的镜像、走个本地rpc等。。

反过来想，只要程序瓶颈还卡在计算上，就有机会用python写一版至少吞吐上可以打平cpp的Dynamic Batching。好处是使用会方便很多。

出于个人需要和兴趣，之前基于multiprocess.Queue写过一版Dynamic Batching。但是Queue本身对于延迟的影响非常大，数字比较难看。

最近发现Python 3.8支持了共享内存，用python写了个基于SharedMemory的Dynamic Batching。

跟大家分享一下效果。

## 测试环境

模型Resnet50，输入(N,3,224,224)。使用某云的V100。

## 测试结果

我们先测一下Torch性能上限，好对数据有个基本了解。

然后一步步看不同功能的影响。

对应测试命令：

```
# 生成一个假模型
python fake_resnet50.py
# 测试
python benchmark.py  --no_dynamic_batch --worker_num=N --worker_batch=M
```

### MPS

多进程Torch + MPS。

|进程数量|Batch|Latency|Throughput|
|:-:|:-:|-:|-:| 
| 1  | 1 | 4.54 ms |220.10 pic/s |
| 4  | 1 | 8.05 ms |496.52 pic/s |
| 8  | 1 | 13.97 ms |**572.57 pic/s** |
| 16  | 1 | 28.15 ms |526.42 pic/s |

可以看出MPS是很有效的，没有MPS时，多进程轮占时间片，多个进程吞吐基本也就卡在200多。

加了多进程后，多进程的kernel在同一context下调度。在8的时候达到最高。

### Batching

基于以上数据，再看下Batching的影响。

|进程数量|Batch|Latency|Throughput|
|:-:|:-:|-:|-:| 
| 4  | 1 | 8.05 ms | 496.52 pic/s |
| 1  | 4 | 6.43 ms | 622.07 pic/s |

|进程数量|Batch|Latency|Throughput|
|:-:|:-:|-:|-:| 
| 8  | 1 | 13.97 ms | 572.57 pic/s |
| 1  | 8 | 10.43 ms | 766.93 pic/s |

|进程数量|Batch|Latency|Throughput|
|:-:|:-:|-:|-:| 
| 16  | 1 | 28.15 ms |526.42 pic/s |
| 1  | 4 | 18.03 ms |887.20 pic/s|

可以看到MPS虽然对吞吐有帮助，但是有条件的话，Batching依旧是更好的选择。


### MPS+Batching测Torch上限

在测一下Batch=32(或者其他比较高的数字都可)，看一下torch框架的上限。

|进程数量|Batch|Latency|Throughput|
|:-:|:-:|-:|-:| 
| 1 | 32 | 33.54 ms | 953.60 pic/s |
| 2 | 32 | 56.98 ms | 1123.20 pic/s |
| 3 | 32 | 78.96 ms | **1215.47 pic/s** |
| 4 | 32 | 109.89 ms | 1164.80 pic/s |

即便batch比较大了，但MPS依旧有提升。

### Dynamic Batching

实际应用中，琐碎请求会带来的性能下降。如果对于延迟的要求没有非常苛刻，那么是可以通过牺牲一部分延迟(用来打Batch)，换取更高的吞吐(省钱)。

所以这轮测试的场景是，有N个数据(业务)进程，每个进程数据batch=1，达到MPS+Batching的上限吞吐。

先试一下对上述最大吞吐的case。128个数据(业务)进程，每个进程灌一张图，后台通过共享内存传输数据并打batch。

测试命令：
```
python benchmark.py --worker_num=128 --worker_batch=1 --max_batch_size=32 --model_num=3 --wait_time=0.01
```
|数据(业务)进程|GPU模型进程|Latency|Throughput|
|:-:|:-:|-:|-:| 
|128|3|103.45 ms|1237.33 pic/s|

能够达到极限延迟，但比最理想的情况增加了20%+的延迟。

找个小的场景试一下：
```
python benchmark.py --worker_num=8 --worker_batch=1 --max_batch_size=4 --model_num=2 --wait_time=0.003
```
|数据(业务)进程|GPU模型进程|Latency|Throughput|
|:-:|:-:|-:|-:| 
|8|2|13.04 ms|613.40 pic/s|

跟前面Torch测试的数字对比，可以理解成这case下8个请求进程被分成两组，总体基本能够达到batch=4的吞吐。

### 时间都去哪了？

针对1200+的最大吞吐场景分析了一下:

延迟由 batch + MPS 的 79 ms 增加至 Dynamic Batching 的 103ms.其中，

  * 19ms 左右是拼batch的时间，其中10ms是命令中的等待时间，还有8.3ms的np.concat时间。
  * 分割输出回各数据进程大概用了1ms。
  * 各种队列的等待时间。
  
总的来说没有不太合理的地方，在benchmark里我也把各部分时间收集和打出来了。

## 代码 & 相关说明

原理大概就是这个 [shared_memory sample](./SimpleDBI/shm_sample.py)

测试代码：[benchmark.py](./benchmark.py)

使用样例：[sample.py](./sample.py)
 - 基本跟用pytorch差不多，load+forward。但是：
    * 要指定数据最大尺寸，用来分配shared memory
    * 最后要用一个Run函数启动，因为要提前初始化一些进程变量
    * 需要为模型指定name。当程序涉及到多个模型的时候,数据进程通过name连接到特定的模型进程。

## Konwn issues

multiprocess.shared_memory在回收时，在一些系统下会报leak或已经释放的error/warning，一些系统正常。

错的系统我跑[官方示例](https://docs.python.org/3/library/multiprocessing.shared_memory.html)也有错。所以还不好判断是什么原因。如果觉得可以忍又不想烦可以用下面的命令禁掉。
```
export PYTHONWARNINGS=ignore
```

### 最后
If **有人感兴趣** and **我有时间** ：
  - 支持一下TensorRT/TensorCore FP16，以及某个特定版本的TF。
  - 画一下施工图纸。虽然源码不长(<1000行)，但各种进程和通信还是有点多的(所以我还没画..)，有个架构图可以帮助想看了解的同学理解。
  - 输出还没有全用shared memory(主要是我懒)，所以大输出模型的 吞吐/延迟 会受到数据拷贝的影响。可以改进。。。
