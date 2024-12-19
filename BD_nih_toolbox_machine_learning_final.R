# NIH TOOLBOX EMOTION BATTERY (NIHTB-EB) AND SUICIDAL BEHAVIOR IN BIPOLAR I----

#### Note: The same machine learning code shown below was used for 3 sets of features:
#### (1) NIHTB-EB measures, sex, and age [shown here]
#### (2) MADRS items, YMRS items, sex, and age
#### (3) NIHTB-EB measures, MADRS items, YMRS items, sex, and age

## Load required packages ----
require(tidyverse)
require(ggprism)
require(randomForest) 
require(randomForestSRC)
require(caret)
require(nestedcv)
require(ggrepel)
require(DescTools)

## Import customized functions ----
source("updated_functions_nih_toolbox_project.R")
## Import customized caret method ----
source("caret_custom_rf_v2.R")

## Read in the prepared data ----
emotion_toolbox_bipolar_only <- read.csv("nih_toolbox_bd_only.csv") %>%
  mutate(suicide_attempt_group = factor(suicide_attempt_group),
         sex = factor(sex),
         age_in_years = as.numeric(age_at_nih_tool))  %>%
  mutate(across(contains("toolbox"), ~as.numeric(.))) %>%
  select(contains("toolbox"), suicide_attempt_group, sex, age_at_nih_tool) 


## Run nested cv on NIHTB-EB measures, age, and sex ----
grid_rf_tb <- expand.grid( mtry=(seq(2, sqrt(ncol(emotion_toolbox_bipolar_only)-1), 1) ))

y_nest_tb <- emotion_toolbox_bipolar_only$suicide_attempt_group
x_nest_tb <- emotion_toolbox_bipolar_only %>% select(-suicide_attempt_group)

set.seed(123, "L'Ecuyer-CMRG")
nested_cv_tb <- nestcv.train(
  y=y_nest_tb,
  x=x_nest_tb,
  method = rf_v2,
  filterFUN = NULL,
  filter_options = NULL,
  weights = NULL,
  balance = NULL,
  balance_options = NULL,
  outer_method = "cv",
  n_outer_folds = 5,
  n_inner_folds = 5,
  outer_folds = NULL,
  inner_folds = NULL,
  pass_outer_folds = FALSE,
  cv.cores = 1,
  trControl = NULL,
  tuneGrid = grid_rf_tb,
  savePredictions = "final",
  outer_train_predict = TRUE,
  finalCV = FALSE,
  na.option = "pass",
  verbose = TRUE,
  importance=TRUE
)

## Get Model Metrics ----

### overall accuracy ----
summary(nested_cv_tb) 

### final variables ----
nested_cv_tb$final_vars

### ROC plots -----
plot(nested_cv_tb$roc)
plot(nested_cv_tb$roc)
plot((innercv_roc(nested_cv_tb)))

### More Performance metrics ----
testy_toolbox <- as.numeric(as.character(recode_factor(nested_cv_tb$output$testy, "Bipolar_No"=0, "Bipolar_Yes"=1)))
predy_toolbox <- as.numeric(as.character(recode_factor(nested_cv_tb$output$predy, "Bipolar_No"=0, "Bipolar_Yes"=1)))
DescTools::BrierScore(resp=testy_toolbox, pred=predy_toolbox, scaled=FALSE)
caret::confusionMatrix(data=nested_cv_tb$output$predy, reference=nested_cv_tb$output$testy,
                       mode = "everything",
                       positive="Bipolar_Yes")

### customized cv_varImp() values for accuracy ----
cv_varImp(nested_cv_tb, metric_imp="accuracy", final=FALSE)

### customized plot_var_stability() for both md and accuracy

plot_toolbox_ml_accuracy_noscaling <- plot_var_stability(nested_cv_tb, metric_imp="accuracy", final = FALSE, final_fitting=FALSE) +
  labs(x="Mean Decrease in Accuracy with Permutation") +
  theme(legend.position="None")

plot_toolbox_ml_accuracy_noscaling

ggsave("plot_toolbox_accuracy_noscaling.pdf",plot_toolbox_ml_accuracy_noscaling, units="in", width=6.5, height=4.5)


plot_toolbox_ml_md_noscaling <- plot_var_stability(nested_cv_tb, metric_imp="md", final = FALSE, final_fitting=FALSE) +
  coord_cartesian(xlim=c(2,6)) +
  labs(x="Mean Minimal Depth") +
  theme(legend.position="None")

ggsave("plot_toolbox_md_noscaling_ml.pdf",plot_toolbox_ml_md_noscaling, units="in", width=6.5, height=4.5)


### Create graph of ranked importance by both accuracy and md, with direction marked ----

directionality_pairwise <- data.frame(variable = sort(names(emotion_toolbox_bipolar_only[-18])),
                                      direction_pairwise = c("No Difference", #age
                                                             "No Difference", #anger-affect
                                                             "Up in BD-I Attempt+", #anger-hostility
                                                             "Up in BD-I Attempt+", #anger-physical
                                                             "Down in BD-I Attempt+", #emotion support
                                                             "Trending Up in BD-I Attempt+", #fear-affect
                                                             "Trending Up in BD-I Attempt+", #fear-somatic
                                                             "Down in BD-I Attempt+", #friendship
                                                             "Trending Down in BD-I Attempt+", #gen life satisfaction
                                                             "Trending Down in BD-I Attempt+", #instrumental support
                                                             "Up in BD-I Attempt+", #loneliness
                                                             "No Difference", #meaning
                                                             "No Difference", #perceived hostility
                                                             "Up in BD-I Attempt+", #perceived rejection
                                                             "Up in BD-I Attempt+", #perceived stress
                                                             "No Difference", #positive affect
                                                             "Up in BD-I Attempt+", #sadness
                                                             "Down in BD-I Attempt+", #self-efficacy,
                                                             "Different in BD-I Attempt+" #sex
                                      ))

#### prepare data frames for variable importance of outer cv accuracy and md ----
cv_varImp_accuracy <- cv_varImp(nested_cv_tb, metric_imp="accuracy", final=FALSE)
cv_varImp_md <- cv_varImp(nested_cv_tb, metric_imp="md", final=FALSE)


cv_varImp_accuracy2 <- cv_varImp_accuracy %>%
  as.data.frame(.) %>%
  rowwise() %>%
  mutate(mean_accuracy = mean(c(V1,V2,V3,V4,V5)),
         sd_for_mean_accuracy = sd(c(V1,V2,V3,V4,V5)),
         sem_for_mean_accuracy = sd_for_mean_accuracy/sqrt(5)) %>%
  select(mean_accuracy, sd_for_mean_accuracy, sem_for_mean_accuracy) %>%
  cbind(., variable=row.names(cv_varImp_accuracy))


cv_varImp_md2 <- cv_varImp_md %>%
  as.data.frame(.) %>%
  rowwise() %>%
  mutate(mean_md = mean(c(V1,V2,V3,V4,V5)),
         sd_for_mean_md = sd(c(V1,V2,V3,V4,V5)),
         sem_for_mean_md = sd_for_mean_md/sqrt(5)) %>%
  select(mean_md, sd_for_mean_md, sem_for_mean_md) %>%
  cbind(., variable=row.names(cv_varImp_md))


importance_cv <- cv_varImp_accuracy2 %>%
  full_join(cv_varImp_md2) %>%
  select(variable, mean_accuracy, sd_for_mean_accuracy, sem_for_mean_accuracy,
         mean_md, sd_for_mean_md, sem_for_mean_md) %>%
  mutate(variable_name = recode_factor(variable, 
                                       age_in_years = "Age",
                                       sex = "Sex",
                                       toolbox_anger_affect_CAT = "Anger-affect",
                                       toolbox_anger_hostile = "Anger-hostility",
                                       toolbox_anger_physical = "Anger-physical",
                                       toolbox_emotion_support = "emotion support",     
                                       toolbox_fear_affect_CAT = "Fear-affect",
                                       toolbox_fear_somatic_arouse = "Fear-somatic",
                                       toolbox_friendship = "Friendship",          
                                       toolbox_general_life_sat_CAT = "Gen. life satisfaction",
                                       toolbox_instrument_support = "Instr. support",
                                       toolbox_loneliness = "Loneliness",          
                                       toolbox_meaning_CAT = "Meaning",
                                       toolbox_perceiv_hostile = "Perceived hostility",
                                       toolbox_perceiv_reject = "Perceived rejection",
                                       toolbox_perceiv_stress = "Perceived stress",
                                       toolbox_pos_affect_CAT = "Positive affect",
                                       toolbox_sad_CAT = "Sadness",           
                                       toolbox_self_effic_CAT = "Self-efficacy"  
  )) %>%
  full_join(directionality_pairwise) %>%
  mutate(direction_pairwise = factor(direction_pairwise)) %>%
  mutate(md_rank = dense_rank(mean_md),
         accuracy_rank = dense_rank(desc(mean_accuracy)))


plot_toolbox_ml_acc_vs_md_noscaling_pairwise <- ggplot(data=importance_cv, aes(y=mean_md, x=mean_accuracy, 
                                                                               color=direction_pairwise)) +
  geom_label_repel(aes(label=accuracy_rank), max.overlaps=40,
                   label.padding=0.1, force=0.00022, fontface="bold") +
  scale_color_manual(values= c("Down in BD-I Attempt+" = "blue",
                               "Trending Down in BD-I Attempt+" = "lightblue",
                               "Up in BD-I Attempt+" = "red", 
                               "Trending Up in BD-I Attempt+" = "pink",
                               "No Difference" = "grey",
                               "Different in BD-I Attempt+" = "purple")) + 
  labs(x="Outer CV Folds' Mean Decrease in Accuracy",
       y="Outer CV Folds' Mean Minimal Depth",
       color="directionality_pairwise") +
  coord_cartesian(xlim=c(-0.003, 0.02)) +
  scale_x_continuous(breaks=c(0, 0.005, 0.01, 0.015, 0.02)) +
  theme_prism() +
  theme(legend.position = "None") 


ggsave("plot_toolbox_acc_vs_md_noscaling_ml_pairwise.pdf",plot_toolbox_ml_acc_vs_md_noscaling_pairwise, units="in", width=6.5, height=4.5)

