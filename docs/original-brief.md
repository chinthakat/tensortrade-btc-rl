# Original brief

The requirements below are the developer's own notes, written before any code
existed. They are reproduced verbatim, typos included, because they explain
several design decisions that otherwise look arbitrary — the interactive
prompts, the CSV trade log and its exact columns, the multi-episode support,
and the TensorTrade influence that gave the repository its name.

Not everything asked for here was built. Funding/swap costs in particular are
still not charged by the simulator; see the Status section of the main
[README](../README.md).

```text
Important points:
- The code should be written in Python.
Should be focused on Binance futures trading on cryptocurrency
customisable leverage should be used 
risk amangement and liquidity management is a must
going short should consider the swap fees
use TensorTrade or components from TensorTrade or code influance from TensorTrade must be used.
important input data such as, data files location, name, no of training steps, leverage, the model to be used ( if there are multiple models ) should be prompted by the training script so user can select.
data file format is similar to below
    ,open,high,low,close,volume,timestamp
    0,42313.9,42535.0,42289.6,42532.5,3531.294999999999,1704067200
    1,42532.4,42603.2,42449.1,42458.5,2245.947,1704068100

action log in a csv format must be generated. 
example:
trade_id,training_step,training_iteration,entry_datetime,close_datetime,side,entry_action,entry_price,close_price,net_pnl,close_reward,entry_net_worth,close_net_worth,trade_duration_hours,status,win_loss,position_size,fees_paid,stop_loss_price,take_profit_price,close_reason
TRADE_00000,0,0,01/01/2024 05:00,01/01/2024 06:15,LONG,BUY,42389.3,42349.6,-293.2684925205257,-0.1785620545750986,99900.0,99515.11607882177,1.25,CLOSED,LOSS,2.3567268154935324,199.7064379454249,42162.21243115249,48747.695,MANUAL

multiple episodes must be supported
same model should be able to re-use for training and must be prompted to select
status bar should be displayed white trainig
```

---

## A note on what used to follow this

Earlier commits appended a long formal document to this brief — *A Framework for
Deep Reinforcement Learning in Cryptocurrency Futures Trading*, roughly 460 lines
of numbered Parts and Sections written in a third-party research register. It is
plainly not the same author as the notes above, and its provenance could not be
established, so it is not published here and is not covered by this
repository's licence. It remains in the git history if you need it.
