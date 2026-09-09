# V3 compatibility-risk search

隔离的真实在线GPU实验，不修改 `OCR-DOTA-V3`。严格比较五个compatibility来源：D、E、O、EO、DO。

定义：`D=1-cos²`；`E=clip(旧semantic leakage,0,1)`；
`O=0.5*relu(max_other_sim-sim_k)`；`EO=(E+O)/2`；`DO=(D+O)/2`；
`compatibility=exp(-risk/0.15)`。每个候选只生成一个compatibility张量，并同时供原V3
prediction prior和update gate使用，固定`prediction_strength=0.1, update_power=1`。

正式运行前固定执行32样本smoke；正式输出支持严格身份断点恢复、STOP、PidLock、JSONL fsync和冷启动复验。
