# ==========================
# R Shiny: Iris ML Demo App
# ==========================

# 套件安裝與載入
if(!require(shiny)) install.packages("shiny")
if(!require(caret)) install.packages("caret")
if(!require(randomForest)) install.packages("randomForest")
if(!require(ggplot2)) install.packages("ggplot2")

library(shiny)
library(caret)
library(randomForest)
library(ggplot2)

set.seed(123)

# ----------- 模型建立區 -----------
data(iris)

ctrl <- trainControl(method = "cv", number = 10)
model <- train(
  Species ~ ., 
  data = iris,
  method = "rf",
  trControl = ctrl
)

accuracy <- mean(model$resample$Accuracy)
importance <- varImp(model)

# ----------- UI 介面 -----------
ui <- fluidPage(
  titlePanel("🌸 Iris Species Prediction — Random Forest Model"),
  
  sidebarLayout(
    sidebarPanel(
      h4("Adjust the Feature Values"),
      sliderInput("sepal_length", "Sepal Length:", 
                  min = 4, max = 8, value = 5.8, step = 0.1),
      sliderInput("sepal_width", "Sepal Width:", 
                  min = 2, max = 4.5, value = 3.0, step = 0.1),
      sliderInput("petal_length", "Petal Length:", 
                  min = 1, max = 7, value = 4.0, step = 0.1),
      sliderInput("petal_width", "Petal Width:", 
                  min = 0.1, max = 2.5, value = 1.3, step = 0.1),
      hr(),
      h4("Model Summary"),
      p("Algorithm: Random Forest"),
      p("Validation: 10-fold Cross Validation"),
      p(paste("Average Accuracy:", round(accuracy, 3))),
      hr(),
      h4("Feature Importance"),
      plotOutput("importance_plot", height = "250px")
    ),
    
    mainPanel(
      h3("🌼 Predicted Species"),
      verbatimTextOutput("prediction"),
      hr(),
      h4("Model Description"),
      p("This Random Forest model was trained on the classic iris dataset using 10-fold cross-validation. 
         The model evaluates feature combinations to classify flowers into three species:
         Setosa, Versicolor, and Virginica."),
      p("You can adjust the feature values using the sliders on the left, 
         and the app will instantly predict the species along with model accuracy information.")
    )
  )
)

# ----------- Server 邏輯 -----------
server <- function(input, output) {
  
  # 即時計算預測結果
  prediction <- reactive({
    newdata <- data.frame(
      Sepal.Length = input$sepal_length,
      Sepal.Width  = input$sepal_width,
      Petal.Length = input$petal_length,
      Petal.Width  = input$petal_width
    )
    predict(model, newdata)
  })
  
  # 顯示預測結果
  output$prediction <- renderText({
    paste("Predicted Species:", prediction())
  })
  
  # 繪製特徵重要性圖
  output$importance_plot <- renderPlot({
    ggplot(importance, aes(x = reorder(rownames(importance$importance),
                                       importance$importance[,1]),
                           y = importance$importance[,1])) +
      geom_bar(stat = "identity", fill = "#B2182B") +
      coord_flip() +
      labs(x = "Feature", y = "Importance", 
           title = "Variable Importance (Random Forest)") +
      theme_minimal()
  })
}

# 啟動 Shiny App
shinyApp(ui = ui, server = server)
