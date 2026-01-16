# Softmax Unit Design Notes

## *A_High-Speed_and_Low-Complexity_Architecture_for_Softmax_Function_in_Deep_Learning*

- institution: Nanjing University.
- published: 2018 IEEE Asia Pacific Conference on Circuits and Systems (APCCAS)
- link: [IEEE Xplore](https://ieeexplore.ieee.org/document/8605654)

### 关于本文的定点小数和组里项目中的浮点数(16bit/32bit)的问题

- 回顾浮点数的表示法：![浮点数表示法](./images/floating_rep.png)
    - 参考：[单精度浮点数(float32)存储与表示方式](https://zhuanlan.zhihu.com/p/632347955)
        - 1 bit 符号位
        - 8 bits 指数位 (exponent)
        - 23 bits 尾数位 (mantissa)
    - 对于16bit浮点数(half-precision floating point, float16)：
        - 1 bit 符号位
        - 5 bits 指数位 (exponent)
        - 10 bits 尾数位 (mantissa)

- 如何把文章中的定点数计算映射到浮点数计算（以32位浮点数为例）？
    - ps：对softmax而言，对输入施加线性偏移是安全的(如把$x_i^{'} = x_i - x_{max}$不影响输出结果），因此可以通过调整偏移来避免溢出/下溢, 或者进行精度提升)
    1. $u_i + v_i = y_i · log_2e$, 把后者分解为整数+小数部分
        - 想要直接分解为两个浮点数是极其困难的，过不我们换一个思路就很简单，实际上我们要求的是$2^{v_i} << u_i \approx (0.94275 + v_i) << u_i = (1+v_i-0.05725) << u_i$, 这样，我们其实只需要把$y_i · log_2e$按照其$exp = exponent- 127(for normalized floating point number)$手动添加小数点，拆分成：
            1. 整数（下取整）
            2. 小数（由于整数是下取整的，所以小数部分大于0）
            - 具体的切法：
                - case1：yi·log2e >= 0 --- 很简单...
                - case2：yi·log2e < 0 --- 先按照正数切分，然后整数部分取反加1（-x = ~x + 1）再减1（---综合下来等价于整数部分取反），小数部分 = 1 - 小数部分(整数减法 -- 就是把小数部分取反加1再加1，所谓加一，就是`{1, lenth_of_frac_bits{1'b0}}`）)
        - 整数直接加在浮点数的指数位上，小数通过
        ```verilog
            wire [31:0] out1 = {1'b0, // sign bit
                              8'd127, // 1+vi <= 2, so exponent = 127 + 0
                              frac_part // frac part 是通过上面写到的手动添加小数点后，取小数点后面的数得到的。
                             };
        ```
        - 所以结果就是：
        ```verilog
            wire [31:0] out2 = out1 - （浮点数减法）32'h3D_6A_7E_F9;  // 0.05725的float32表示
            wire [31:0] out3 = {1'b0, // sign bit
                              int_part + out2[30:23], // int_part是上面拆分出来的整数部分, 补码有符号数加法
                              out2[22:0] // frac part 是通过上面写到的手动添加小数点后，取小数点后面的数得到的。
                             };
        ```
    2. $F = 2^w · k$分解，这样$log_2F = w + log_2k$, 再进一步用$log_2k \approx k - 0.94275$近似求解。
        - 这一步浮点数对应到$w$和$k$很简单，分规格数和非规格数处理就行（非规格数可以直接置0吗？）。

## *Design and Implementation of an Approximate  Softmax Layer for Deep Neural Networks*

- institution: 
    - College of Electronic and Information Engineering, Nanjing University of Aeronautics and Astronautics
    - Department of Electrical and Computer Engineering, Northeastern University
- published: 2020 IEEE International Symposium on Circuits and Systems (ISCAS)
- link: [IEEE Xplore](https://ieeexplore.ieee.org/document/9180870)

### 关于浮点数定点化的问题
- **在上一篇文章中，我总是想把浮点数如何凑出文章中用定点数表示的部分，但实际上根本不用这么复杂啊！**，All you need is 做一个浮点->定点转换即可!
    - 只需要按照exponent的值把`1.mantissa`左移或者右移, 之后再根据我们定点数的设计（位宽/小数点位置）进行截断即可。
    - 但是这个硬件开销...
        - 每一个浮点数输入（如果要实现并行处理）：
            - 需要一个浮点数的解码器，把浮点数拆成符号位/指数位/尾数位
            - 需要一个可变长的移位器 (barrel shifter) 来做左移/右移
            - 需要一个定点数的截断模块
            - 开销 *= n（如一个n分类问题）
        - 如果不要求输入并行转换
            - 上述各一个就行。
            - 串行执行？
                - 通过一个计数器FSM来控制每次处理哪个输入即可。
                - 通过拉高一个`valid`信号来告master端数据已经准备好，可以开始处理下一个输入了。
                - AXI协议的话，假设一个burst最大传输位宽是512位，也就是16个32位floating point数据。
                - 无法流水线化？
    - 主要是现在还不知道项目的具体要求，所以感觉设计空间太大了，只能先把思路理清楚，等有具体要求再做设计。

### 关于文中section 4最后那一段话：可以参考gemini的解释:
- 这段文字主要描述了论文中提出的 **除法单元（DIVU）** 相对于传统对数除法器的优化策略。它解释了作者如何通过简化计算步骤和近似处理来降低硬件复杂度，同时保持神经网络所需的精度。

以下是对这段话的详细解读：

#### 1. 传统对数除法器的问题
> *"In a conventional logarithmic divider, every input bit is calculated in the division operation, the accuracy decreases due to the approximation in the logarithmic conversion."*

*   **背景：** 传统的对数除法器通常会将输入的每一个比特都参与到对数转换中，试图将除法 $A/B$ 转换为减法 $\log(A) - \log(B)$。
*   **问题：** 这种转换通常是近似的（例如使用 $log_2(1+x) \approx x$）。如果对每一个比特位都进行这种近似计算，或者试图对全位宽数据进行对数变换，累积的误差会导致精度下降，且硬件资源消耗较大。

#### 2. 提出的设计方案：位宽设置与简化
> *"In the proposed design, the divisor data width has been set to 8-bits, while the dividend is 16-bits."*

*   **配置：** 作者针对Softmax的特性定制了位宽：
    *   **被除数（Dividend, $e^{x_i}$）：** 保持 **16-bit** 精度。这是单个神经元的输出，需要较高的动态范围。
    *   **除数（Divisor, $\sum e^{x_j}$）：** 设置为 **8-bit** 精度。这是所有指数的和，作为一个归一化因子，在推理阶段对其精度的敏感度相对较低。

#### 3. 核心算法：省略余数转换，直接叠加
> *"Therefore, the remainder is omitted in the logarithmic conversion, and added to the final results to achieve an approximate operation."*

这是这段话最难理解的部分，需要结合公式 (8) 和 (9) 来解释：

$$ \log_2(\frac{N_1}{N_2}) \approx (S_1 - S_2) + (F_1 - F_2) $$

*   **“Logarithmic conversion” (对数转换)：** 正常的对数除法需要计算 $\log_2(F)$（其中 $F$ 是归一化后的小数部分/尾数）。
*   **“Remainder is omitted” (省略余数转换)：** 这里的“Remainder”指的是在提取最高有效位（Leading One Detection）后剩下的**小数部分（Mantissa/Fractional parts, 即公式中的 $F$）**。
    *   作者**没有**把这个小数部分 $F$ 拿去查表或进行复杂的对数运算。
    *   而是直接使用了线性近似：$\log_2 F \approx F$（文中公式9）。这意味着他们直接“省略”了对小数部分的对数变换步骤。
*   **“Added to the final results” (加到最终结果)：**
    *   运算变成了简单的线性操作：直接算出整数阶码的差 $(S_1 - S_2)$。
    *   然后将小数部分的差 $(F_1 - F_2)$ 直接**加**到前面的结果上。

#### 总结
这段话的意思是：
为了避免传统对数除法器因全位宽近似带来的高复杂度和误差，作者将除数缩减为8位，并且在计算 $\log$ 时，**不再对小数部分（余数）进行复杂的对数转换，而是直接将其作为线性数值加到最终的减法结果中**。这使得整个除法运算被转化为简单的“移位（提取阶码 $S$）”和“减法（$S$相减，$F$相减）”操作，极大地节省了硬件资源。  