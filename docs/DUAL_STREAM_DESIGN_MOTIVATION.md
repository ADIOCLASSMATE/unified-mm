# 双流架构的设计动机

我们关注统一多模态模型中的一个结构性观察：**共享同一个 backbone 时，文本与图像仍可能采用不同的预测方式。** 例如，[Transfusion](https://arxiv.org/html/2408.11039v1) 结合文本 next-token prediction 与图像 diffusion，[Show-o2](https://arxiv.org/html/2506.15564v2) 结合文本自回归与图像 flow matching。

文本 next-token prediction 从已知前缀的最后一个位置读出，预测下一个 token；图像 diffusion / flow matching 则在目标图像位置结合带噪状态，预测该位置的噪声或速度。两者都根据上下文预测未知内容，但预测位置与输入内容的对应方式不同。

| 建模方式 | 用于预测的状态 | 预测目标 |
| --- | --- | --- |
| 文本 next-token prediction | 前一位置的内容状态 | 下一个文本 token |
| 图像 diffusion / flow matching | 目标位置的状态，结合该位置的带噪输入 | 该图像位置的噪声或速度 |

这一观察引出我们的设计选择：**显式区分已知内容与待预测目标，让文本和图像都在目标位置查询上下文。** 在此基础上，我们进一步区分各模态的生成次序，以及 backbone 与图像 flow head 的职责。

我们采用共享参数的 content/query 双流。Content stream 编码已知内容，并提供上下文 K/V；query stream 表示待预测的位置，从允许的上下文中形成预测状态。内容与查询的区分借鉴了 [XLNet 的 two-stream attention](https://arxiv.org/html/1906.08237v2)，并用于统一文本与连续图像的条件预测。

在本项目的 B 架构中，content 按 `sigma_kv <= sigma_q` 读取，query 按 `sigma_kv < sigma_q` 读取。文本 query 在位置 i 预测 token i，因而消除文本预测的一位 shift；严格的 query 可见性避免读取目标内容。图像也采用目标位置上的 query，并交由 flow head 处理连续生成。两种模态保留各自的文本分类头和图像 flow head。

**统一建模应允许不同模态采用各自的生成顺序。** 文本具有自然的从左到右顺序，图像的二维空间位置则可以采用不同的遍历方式。我们将目标的物理位置与生成次序分开：位置编码说明“预测哪里”，sigma 与可见性约束说明“哪些内容已经可用”。Query 因而可以在任意选定的目标位置查询已知上下文。文本保持从左到右的自然顺序；图像可选择随机、空间覆盖等顺序，按选定次序依赖已生成内容，无需绑定到从左到右的栅格自回归。当前实现支持 random、spatial_halton、spatial_uniform 等策略，见[生成顺序比较](B_ORDER_SWEEP.md)。

**我们主张 backbone 专注于跨模态关联与上下文建模，图像的生成时间步迭代由 flow head 承担。** Backbone 根据已知文本、已知图像内容和目标位置形成条件表示；flow head 结合这些条件、当前 noisy latent 和时间 t，求解目标图像位置的连续状态演化。这一分工将上下文建模与数值积分分开，使同一上下文条件可以在多个 flow 时间步之间复用。

在 B 中，backbone query 使用 learned mask 初始化；noisy latent 与时间输入 flow head，backbone 的 query/content hidden 分别提供对应的条件。同一目标位置、已知上下文不变时，backbone 条件在 flow 时间步迭代中保持固定。当前目标生成完成并进入下一位置时，backbone 更新已知内容与缓存，形成下一目标的条件。生成次序的推进与同一目标内部的时间步迭代由此分别处理。

文本与图像采用一致的“已知内容 → 目标 query → 模态输出头”组织方式，同时保留适合各模态的生成顺序，并将图像的连续生成过程放在 flow head 中。具体结构和对照组见[实验定义](EXPERIMENTS.md)。

论文中的 motivation 可表述为：

> 现有一类统一多模态模型在共享 backbone 中结合文本 next-token prediction 与图像 diffusion 或 flow matching。我们观察到，两者虽然都根据上下文预测未知内容，却采用不同的预测位置约定。为统一这两种预测方式，我们引入共享参数的 content/query 双流，将已知内容的编码与目标位置的查询显式分开。文本与图像均通过各自目标位置的 query 完成预测，文本的一位 shift 因而被消除，同时通过可见性约束保持因果性。
>
> 我们认为，统一模型应保留各模态适合的生成次序。通过将物理位置与生成顺序解耦，文本可以遵循从左到右的自然顺序，图像则可以按随机或空间覆盖等次序生成，无需沿用文本式的栅格顺序。进一步地，我们主张 backbone 专注于跨模态关联与上下文建模，由 flow head 承担图像连续状态的时间步迭代。对于给定的目标位置与已知上下文，backbone 提供固定条件，flow head 据此完成生成；上下文随生成次序推进而更新。这使统一的条件建模、模态特定的生成顺序和图像连续生成形成明确分工。

以上构成架构的设计动机。C 的文本 AR、D 的 Dynamic-XT、E 的 sequential 等现有统一训练消融与下游评测用于衡量相应设计选择的实际效果。
