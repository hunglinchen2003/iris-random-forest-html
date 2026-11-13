# 鳶尾花隨機森林獨立網頁

把 `03_Machine_learning.R` 的 Shiny app 轉成單一 HTML。瀏覽器裡的預測使用 R 匯出的同一批隨機森林；走樹與投票規則對齊 `randomForest::predict()`。

網頁：<https://hunglinchen2003.github.io/iris-random-forest-html/>

直接打開 `index.html` 也可以離線使用。頁面內有轉換過程、計算公式，以及與 R 的核對結果。

重新產生這個網頁：

```bash
Rscript export_model.R
```

需要已安裝 `caret`、`randomForest`、`jsonlite`、`V8`、`ggplot2`。腳本會先核對 JavaScript 與 R 的投票向量，通過後才覆寫 `index.html`。
