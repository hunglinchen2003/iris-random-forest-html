# Export the 03_Machine_learning.R random forest into index.html.
#
# Training below is the same call as the Shiny script: set.seed(123),
# 10-fold CV, caret method "rf". The browser does not refit the forest.
# It walks the trees from randomForest::getTree() with predict.js, and
# this script checks that JavaScript against randomForest::predict().

args_cmd <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_cmd, value = TRUE)
root <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
} else {
  getwd()
}

suppressPackageStartupMessages({
  library(caret)
  library(randomForest)
  library(jsonlite)
  library(V8)
  library(ggplot2)
})

message("Training the caret random forest (same call as 03_Machine_learning.R)...")
set.seed(123)
data(iris)

ctrl <- trainControl(method = "cv", number = 10)
model <- train(
  Species ~ .,
  data = iris,
  method = "rf",
  trControl = ctrl
)

rf <- model$finalModel
classes <- as.character(rf$classes)
accuracy <- mean(model$resample$Accuracy)
vnames <- rownames(rf$importance)
if (is.null(vnames)) vnames <- colnames(iris)[1:4]

message("Selected mtry = ", rf$mtry, ", ntree = ", rf$ntree,
        ", CV accuracy = ", format(accuracy, digits = 10))

f64_b64 <- function(x) {
  if (!length(x)) return("")
  con <- rawConnection(raw(0), "w")
  writeBin(as.double(x), con, size = 8, endian = "little")
  val <- rawConnectionValue(con)
  close(con)
  gsub("\\s+", "", base64_enc(val))
}

extract_trees <- function(forest, nclass) {
  lapply(seq_len(forest$ntree), function(k) {
    tr <- getTree(forest, k = k, labelVar = FALSE)
    status_terminal <- tr[, 5] == -1
    split_terminal <- tr[, 3] == 0
    if (!identical(as.logical(status_terminal), as.logical(split_terminal))) {
      stop("Tree ", k, ": terminal status does not match split var == 0")
    }
    internal <- which(!status_terminal)
    if (length(internal)) {
      if (any(tr[internal, 1] < 1 | tr[internal, 1] > nrow(tr))) {
        stop("Tree ", k, ": left daughter out of range")
      }
      if (any(tr[internal, 2] < 1 | tr[internal, 2] > nrow(tr))) {
        stop("Tree ", k, ": right daughter out of range")
      }
      if (any(tr[internal, 3] < 1 | tr[internal, 3] > length(vnames))) {
        stop("Tree ", k, ": split variable out of range")
      }
    }
    pred <- as.integer(tr[, 6])
    terminal <- which(status_terminal)
    if (!length(terminal) || any(pred[terminal] < 1L | pred[terminal] > nclass)) {
      stop("Tree ", k, ": leaf class id out of range")
    }
    list(
      L1 = as.integer(tr[, 1]),
      R1 = as.integer(tr[, 2]),
      L = as.integer(ifelse(tr[, 1] == 0, -1L, tr[, 1] - 1L)),
      R = as.integer(ifelse(tr[, 2] == 0, -1L, tr[, 2] - 1L)),
      V = as.integer(tr[, 3]),
      S = as.numeric(tr[, 4]),
      C = pred
    )
  })
}

message("Reading ", rf$ntree, " trees with getTree()...")
trees <- extract_trees(rf, length(classes))

walk_one <- function(tr, x) {
  node <- 1L
  guard <- 0L
  while (tr$V[node] != 0L) {
    guard <- guard + 1L
    if (guard > length(tr$V)) stop("R walker did not reach a leaf")
    f <- tr$V[node]
    if (x[[f]] <= tr$S[node]) node <- tr$L1[node] else node <- tr$R1[node]
  }
  tr$C[node]
}

message("Checking getTree() against predict(..., predict.all = TRUE)...")
check_df <- iris[c(1, 51, 101, 75, 120), vnames, drop = FALSE]
pa <- predict(rf, check_df, predict.all = TRUE)
indiv <- matrix(as.character(pa$individual), nrow = nrow(check_df), ncol = rf$ntree)
for (i in seq_len(nrow(check_df))) {
  x <- as.numeric(check_df[i, vnames])
  for (k in seq_len(rf$ntree)) {
    got <- classes[walk_one(trees[[k]], x)]
    if (!identical(got, indiv[i, k])) {
      stop("Tree walk mismatch at row ", i, " tree ", k,
           ": walker ", got, " vs predict.all ", indiv[i, k])
    }
  }
}
message("Per-tree walk matches predict.all on 5 rows x ", rf$ntree, " trees.")

imp <- varImp(model)
raw_imp <- importance(rf)
if (is.null(dim(raw_imp))) {
  raw_vec <- as.numeric(raw_imp)
  names(raw_vec) <- names(raw_imp)
} else {
  raw_vec <- as.numeric(raw_imp[, 1])
  names(raw_vec) <- rownames(raw_imp)
}
if (!all(vnames %in% names(raw_vec))) stop("Importance names do not match the forest")
raw_vec <- raw_vec[vnames]
scaled <- as.numeric(imp$importance[, 1])
names(scaled) <- rownames(imp$importance)
if (!all(vnames %in% names(scaled))) stop("varImp names do not match the forest")
scaled <- unname(scaled[vnames])
raw_vec <- unname(raw_vec)

shifted <- raw_vec - min(raw_vec)
scaled_manual <- shifted / max(shifted) * 100
if (!isTRUE(all.equal(as.numeric(scaled_manual), as.numeric(scaled), tolerance = 0))) {
  stop("Manual varImp scaling does not reproduce caret::varImp() exactly")
}

slider_spec <- list(
  Sepal.Length = list(label = "Sepal Length", zh = "花萼長度", min = 4, max = 8, step = 0.1, value = 5.8),
  Sepal.Width = list(label = "Sepal Width", zh = "花萼寬度", min = 2, max = 4.5, step = 0.1, value = 3.0),
  Petal.Length = list(label = "Petal Length", zh = "花瓣長度", min = 1, max = 7, step = 0.1, value = 4.0),
  Petal.Width = list(label = "Petal Width", zh = "花瓣寬度", min = 0.1, max = 2.5, step = 0.1, value = 1.3)
)

features <- lapply(vnames, function(nm) {
  s <- slider_spec[[nm]]
  if (is.null(s)) stop("Unexpected predictor: ", nm)
  list(
    name = unbox(nm),
    label = unbox(s$label),
    zh = unbox(s$zh),
    min = unbox(s$min),
    max = unbox(s$max),
    step = unbox(s$step),
    value = unbox(s$value)
  )
})

rs <- model$resample
rs <- rs[order(rs$Resample), , drop = FALSE]
folds <- lapply(seq_len(nrow(rs)), function(i) {
  list(
    name = unbox(as.character(rs$Resample[i])),
    Accuracy = unbox(rs$Accuracy[i]),
    Kappa = unbox(rs$Kappa[i])
  )
})

tune <- lapply(seq_len(nrow(model$results)), function(i) {
  list(
    mtry = unbox(as.integer(model$results$mtry[i])),
    Accuracy = unbox(model$results$Accuracy[i]),
    AccuracySD = unbox(model$results$AccuracySD[i]),
    selected = unbox(as.integer(model$results$mtry[i]) == as.integer(model$bestTune$mtry))
  )
})

preset_rows <- c(1, 51, 101)
named_default <- c(Sepal.Length = 5.8, Sepal.Width = 3.0, Petal.Length = 4.0, Petal.Width = 1.3)
presets <- c(
  list(list(
    name = unbox("Shiny 預設值"),
    x = I(unname(named_default[vnames])),
    truth = NULL
  )),
  lapply(preset_rows, function(i) {
    list(
      name = unbox(paste0("訓練第 ", i, " 筆")),
      x = I(unname(as.numeric(iris[i, vnames]))),
      truth = unbox(as.character(iris$Species[i]))
    )
  })
)

train_pred <- predict(model, iris)
tab <- table(
  Predicted = factor(train_pred, levels = classes),
  Actual = factor(iris$Species, levels = classes)
)
confusion <- lapply(seq_along(classes), function(i) I(unname(as.integer(tab[i, ]))))
train_accuracy <- mean(train_pred == iris$Species)

model_obj <- list(
  classes = I(unname(classes)),
  cutoff = I(unname(as.numeric(rf$forest$cutoff))),
  features = features,
  ntree = unbox(as.integer(rf$ntree)),
  mtry = unbox(as.integer(rf$mtry)),
  accuracy = unbox(accuracy),
  accuracyLabel = unbox(as.character(round(accuracy, 3))),
  accuracyText = unbox(format(accuracy, digits = 16)),
  folds = folds,
  tune = tune,
  importance = list(
    features = I(unname(vnames)),
    raw = unbox(f64_b64(raw_vec)),
    scaled = I(unname(as.numeric(scaled)))
  ),
  iris = list(
    values = lapply(vnames, function(nm) I(unname(iris[[nm]]))),
    species = I(unname(as.character(iris$Species)))
  ),
  presets = presets,
  confusion = list(
    classes = I(classes),
    matrix = confusion
  ),
  trainAccuracy = unbox(train_accuracy),
  trainAccuracyLabel = unbox(as.character(round(train_accuracy, 3))),
  trees = lapply(trees, function(tr) {
    list(
      L = I(unname(tr$L)),
      R = I(unname(tr$R)),
      V = I(unname(tr$V)),
      S = unbox(f64_b64(tr$S)),
      C = I(unname(tr$C))
    )
  }),
  versions = list(
    r = unbox(R.version.string),
    caret = unbox(as.character(packageVersion("caret"))),
    randomForest = unbox(as.character(packageVersion("randomForest"))),
    rng = unbox(paste(RNGkind(), collapse = " / ")),
    seed = unbox(123L)
  ),
  nodeCount = unbox(sum(vapply(trees, function(tr) length(tr$V), integer(1)))),
  generated = unbox(format(Sys.time(), "%Y-%m-%d %H:%M %Z"))
)

message("Building the verification set...")
grid <- expand.grid(
  Sepal.Length = seq(4, 8, by = 1),
  Sepal.Width = seq(2, 4.5, by = 0.5),
  Petal.Length = seq(1, 7, by = 1),
  Petal.Width = seq(0.1, 2.5, by = 0.4)
)
set.seed(456)
n_rand <- 200
random_df <- data.frame(
  Sepal.Length = runif(n_rand, 4, 8),
  Sepal.Width = runif(n_rand, 2, 4.5),
  Petal.Length = runif(n_rand, 1, 7),
  Petal.Width = runif(n_rand, 0.1, 2.5)
)
extra_df <- data.frame(
  Sepal.Length = c(5.8, 4, 8),
  Sepal.Width = c(3.0, 2, 4.5),
  Petal.Length = c(4.0, 1, 7),
  Petal.Width = c(1.3, 0.1, 2.5)
)
cases_df <- rbind(
  iris[, vnames, drop = FALSE],
  grid[, vnames, drop = FALSE],
  random_df[, vnames, drop = FALSE],
  extra_df[, vnames, drop = FALSE]
)
cases_df[] <- lapply(cases_df, as.numeric)

cases_json <- toJSON(lapply(seq_len(nrow(cases_df)), function(i) {
  as.numeric(cases_df[i, ])
}), digits = 17, auto_unbox = TRUE)
cases_back <- fromJSON(cases_json)
if (is.list(cases_back) && !is.data.frame(cases_back)) cases_back <- do.call(rbind, cases_back)
cases_back <- as.matrix(cases_back)
if (!identical(as.numeric(cases_back), as.numeric(as.matrix(cases_df)))) {
  stop("Verification inputs did not round-trip through JSON")
}
cases_df <- as.data.frame(cases_back)
names(cases_df) <- vnames

message("Predicting ", nrow(cases_df), " rows in R and in the page's JavaScript...")
rv <- predict(rf, cases_df, type = "vote", norm.votes = FALSE)
rv <- unname(as.matrix(rv[, classes, drop = FALSE]))
r_class <- as.character(predict(model, cases_df))
rf_class <- as.character(predict(rf, cases_df))
message("Class labels that differ between caret and randomForest: ",
        sum(r_class != rf_class),
        " (expected only when votes tie and each predict() draws its own tie break)")

model_json <- toJSON(model_obj, auto_unbox = FALSE, digits = 17, null = "null", na = "null")
model_json <- gsub("<", "\\u003c", model_json, fixed = TRUE)

predict_src <- paste(readLines(file.path(root, "predict.js"), encoding = "UTF-8", warn = FALSE), collapse = "\n")
ctx <- v8()
invisible(ctx$eval(predict_src))
invisible(ctx$eval(paste0("var model = ", model_json, ";")))
invisible(ctx$eval("hydrateModel(model);"))
invisible(ctx$eval(paste0("var cases = ", cases_json, ";")))
invisible(ctx$eval("
var voteRows = new Array(cases.length);
var i, j;
for (i = 0; i < cases.length; i++) voteRows[i] = predictCase(model, cases[i]).votes;
var voteSum = 0;
for (i = 0; i < voteRows.length; i++) {
  for (j = 0; j < voteRows[i].length; j++) voteSum += voteRows[i][j];
}
var jsScaled = scaleImportance(model.importance.rawValues);
var scaledDiff = 0;
for (i = 0; i < jsScaled.length; i++) {
  var d = Math.abs(jsScaled[i] - model.importance.scaled[i]);
  if (d > scaledDiff) scaledDiff = d;
}
var foldAcc = new Array(model.folds.length);
for (i = 0; i < model.folds.length; i++) foldAcc[i] = model.folds[i].Accuracy;
var meanDiff = Math.abs(meanAccuracy(foldAcc) - model.accuracy);
var payload = JSON.stringify({
  votes: voteRows,
  voteSum: voteSum,
  scaledDiff: scaledDiff,
  meanDiff: meanDiff
});
"))
payload <- fromJSON(ctx$get("payload"))
js_votes <- payload$votes
if (is.list(js_votes) && !is.matrix(js_votes)) js_votes <- do.call(rbind, js_votes)
js_votes <- as.matrix(js_votes)
if (!identical(dim(js_votes), dim(rv))) {
  stop("Vote matrix size ", paste(dim(js_votes), collapse = "x"),
       " != ", paste(dim(rv), collapse = "x"))
}
if (!identical(as.numeric(payload$voteSum), as.numeric(nrow(cases_df) * rf$ntree))) {
  stop("JavaScript vote total is not n_cases * ntree")
}
vote_abs <- abs(js_votes - rv)
vote_mismatches <- sum(apply(vote_abs, 1, max) > 0)
if (vote_mismatches > 0) {
  idx <- which(apply(vote_abs, 1, max) > 0)[1]
  message("First vote mismatch at row ", idx)
  print(cases_df[idx, , drop = FALSE])
  print(js_votes[idx, ])
  print(rv[idx, ])
  stop("JavaScript votes do not match randomForest::predict(type = 'vote')")
}

cutoff <- as.numeric(rf$forest$cutoff)
equal_cutoff <- all(cutoff == cutoff[1])
tie <- logical(nrow(js_votes))
det <- character(nrow(js_votes))
tie_bad <- 0L
for (i in seq_len(nrow(js_votes))) {
  if (equal_cutoff) {
    leaders <- classes[js_votes[i, ] == max(js_votes[i, ])]
  } else {
    score <- js_votes[i, ] / cutoff
    leaders <- classes[score == max(score)]
  }
  tie[i] <- length(leaders) > 1
  det[i] <- leaders[1]
  if (tie[i] && !(r_class[i] %in% leaders && rf_class[i] %in% leaders)) tie_bad <- tie_bad + 1L
}
non_tie_bad <- sum(!tie & (det != r_class | det != rf_class))
if (tie_bad > 0 || non_tie_bad > 0) {
  stop("Class decisions disagree with R: non-tie mismatches = ", non_tie_bad,
       ", illegal tie picks = ", tie_bad)
}

message("Votes match on all ", nrow(cases_df), " rows. Ties: ", sum(tie),
        ". Non-tie class mismatches: ", non_tie_bad)

ggplot_note <- tryCatch({
  importance <- varImp(model)
  p <- ggplot(importance, aes(
    x = reorder(rownames(importance$importance), importance$importance[, 1]),
    y = importance$importance[, 1]
  )) +
    geom_bar(stat = "identity", fill = "#B2182B") +
    coord_flip()
  ggplot_build(p)
  "ok"
}, error = function(e) conditionMessage(e))

example_src <- data.frame(
  name = c("Shiny 預設值", "訓練第 1 筆", "訓練第 51 筆", "訓練第 101 筆", "滑桿最小角", "滑桿最大角"),
  Sepal.Length = c(5.8, iris$Sepal.Length[1], iris$Sepal.Length[51], iris$Sepal.Length[101], 4, 8),
  Sepal.Width = c(3.0, iris$Sepal.Width[1], iris$Sepal.Width[51], iris$Sepal.Width[101], 2, 4.5),
  Petal.Length = c(4.0, iris$Petal.Length[1], iris$Petal.Length[51], iris$Petal.Length[101], 1, 7),
  Petal.Width = c(1.3, iris$Petal.Width[1], iris$Petal.Width[51], iris$Petal.Width[101], 0.1, 2.5),
  truth = c(NA, "setosa", "versicolor", "virginica", NA, NA),
  stringsAsFactors = FALSE
)
ex_json <- toJSON(lapply(seq_len(nrow(example_src)), function(i) {
  as.numeric(example_src[i, vnames])
}), digits = 17, auto_unbox = TRUE)
ex_x <- fromJSON(ex_json)
if (is.list(ex_x) && !is.data.frame(ex_x)) ex_x <- do.call(rbind, ex_x)
ex_x <- as.matrix(ex_x)
ex_df <- as.data.frame(ex_x)
names(ex_df) <- vnames
ex_votes <- unname(as.matrix(predict(rf, ex_df, type = "vote", norm.votes = FALSE)[, classes, drop = FALSE]))
ex_class <- as.character(predict(model, ex_df))
examples <- lapply(seq_len(nrow(example_src)), function(i) {
  if (equal_cutoff) {
    leaders <- classes[ex_votes[i, ] == max(ex_votes[i, ])]
  } else {
    score <- ex_votes[i, ] / cutoff
    leaders <- classes[score == max(score)]
  }
  list(
    name = unbox(example_src$name[i]),
    x = I(unname(as.numeric(ex_x[i, ]))),
    votes = I(unname(as.integer(round(ex_votes[i, ])))),
    rSpecies = unbox(ex_class[i]),
    leaders = I(unname(leaders)),
    tie = unbox(length(leaders) > 1),
    truth = if (is.na(example_src$truth[i])) NULL else unbox(example_src$truth[i])
  )
})

if (any(tie)) {
  i <- which(tie)[1]
  leaders <- if (equal_cutoff) {
    classes[js_votes[i, ] == max(js_votes[i, ])]
  } else {
    score <- js_votes[i, ] / cutoff
    classes[score == max(score)]
  }
  examples <- c(examples, list(list(
    name = unbox("驗證集裡的一筆平手"),
    x = I(unname(as.numeric(cases_df[i, ]))),
    votes = I(unname(as.integer(round(js_votes[i, ])))),
    rSpecies = unbox(r_class[i]),
    leaders = I(unname(leaders)),
    tie = unbox(TRUE),
    truth = NULL
  )))
}

verify <- list(
  trainingRows = unbox(nrow(iris)),
  gridRows = unbox(nrow(grid)),
  randomRows = unbox(n_rand),
  extraRows = unbox(nrow(extra_df)),
  totalRows = unbox(nrow(cases_df)),
  voteMismatches = unbox(vote_mismatches),
  nonTieClassMismatches = unbox(non_tie_bad),
  ties = unbox(sum(tie)),
  tieOutsideLeaders = unbox(tie_bad),
  caretMatchesRandomForest = unbox(TRUE),
  treeWalkMatchesPredictAll = unbox(TRUE),
  scaledDiff = unbox(payload$scaledDiff),
  meanDiff = unbox(payload$meanDiff),
  equalCutoff = unbox(equal_cutoff),
  ggplot = unbox(ggplot_note),
  examples = examples,
  gridNote = unbox("Sepal.Length 4 to 8 by 1; Sepal.Width 2 to 4.5 by 0.5; Petal.Length 1 to 7 by 1; Petal.Width 0.1 to 2.5 by 0.4"),
  randomSeed = unbox(456L)
)

verify_json <- toJSON(verify, auto_unbox = FALSE, digits = 17, null = "null", na = "null")
verify_json <- gsub("<", "\\u003c", verify_json, fixed = TRUE)

parsed_model <- fromJSON(model_json, simplifyDataFrame = FALSE)
parsed_v <- parsed_model$trees[[1]]$V
parsed_s <- parsed_model$trees[[1]]$S
parsed_cutoff <- unlist(parsed_model$cutoff)
if (!is.numeric(parsed_v) || length(parsed_v) < 1L) stop("tree nodes were not serialized as arrays")
if (!is.character(parsed_s) || !nzchar(parsed_s)) stop("split points were not serialized")
if (!is.numeric(parsed_cutoff) || length(parsed_cutoff) != length(classes)) {
  stop("cutoff was not serialized as a numeric array")
}

insert_token <- function(html, token, value) {
  parts <- strsplit(html, token, fixed = TRUE)[[1]]
  if (length(parts) != 2L) stop("Placeholder ", token, " was found ", length(parts) - 1L, " times")
  paste0(parts[1], value, parts[2])
}

template_path <- file.path(root, "index.template.html")
template <- paste(readLines(template_path, encoding = "UTF-8", warn = FALSE), collapse = "\n")
html <- template
html <- insert_token(html, "___PREDICT_JS___", predict_src)
html <- insert_token(html, "___MODEL_JSON___", model_json)
html <- insert_token(html, "___VERIFY_JSON___", verify_json)

out_path <- file.path(root, "index.html")
out_con <- file(out_path, open = "wb")
writeBin(charToRaw(enc2utf8(html)), out_con)
close(out_con)
message("Wrote ", out_path)
message("scaledDiff = ", payload$scaledDiff, " meanDiff = ", payload$meanDiff,
        " ggplot = ", ggplot_note)
