# NIH TOOLBOX work ----

require(tidyverse)
require(ggprism)
require(randomForest) 
require(randomForestSRC)
require(caret)
require(nestedcv)
require(ggrepel)

## Import customized functions ----
source("updated_functions_nih_toolbox_project.R")
## Import customized caret method ----
source("caret_custom_rf_v2.R")

## Read in the prepared data ----
emotional_toolbox_bipolar_only <- read.csv("data_emotional_toolbox_BD_groups_only.csv") %>%
  mutate(suicide_attempt_group = factor(suicide_attempt_group),
         sex = factor(sex),
          age_in_years = as.numeric(age_in_years))  %>%
  mutate(across(contains("toolbox"), ~as.numeric(.)))

## Run nested cv ----

grid_rf <- expand.grid( mtry=c(1:sqrt(ncol(emotional_toolbox_bipolar_only)-1)) )

y_nest <- emotional_toolbox_bipolar_only$suicide_attempt_group
x_nest <- emotional_toolbox_bipolar_only %>% select(-suicide_attempt_group)

set.seed(123)
nested_cv <- nestcv.train(
  y=y_nest,
  x=x_nest,
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
  tuneGrid = grid_rf,
  savePredictions = "final",
  outer_train_predict = TRUE,
  finalCV = FALSE,
  na.option = "pass",
  verbose = TRUE,
  importance=TRUE
)
# nested_cv$outer_result[[1]]$fit$finalModel$importance
# randomForestExplainer::measure_importance(nested_cv$outer_result[[1]]$fit$finalModel)
# nested_cv$final_fitting$finalModel$importance
# randomForestExplainer::measure_importance(nested_cv$final_fitting$finalModel)

# ggplot(data=importance_cv, aes(y=mean_min_depth, x=accuracy_decrease)) +
#   geom_point() +
#   geom_label_repel(aes(label=variable_name), max.overlaps=20)


## Get Model Metrics ----

### outer accuracy ----
summary(nested_cv) 

### final variables ----
nested_cv$final_vars

### ROC plots -----
plot(nested_cv$roc)
plot(nested_cv$roc)
plot((innercv_roc(nested_cv)))

### customized cv_varImp() values for accuracy ----
cv_varImp(nested_cv, metric_imp="accuracy", final=FALSE)

### customized plot_var_stability() for both md and accuracy

plot_toolbox_ml_accuracy_noscaling <- plot_var_stability(nested_cv, metric_imp="accuracy", final = FALSE, final_fitting=FALSE) +
  labs(x="Mean Decrease in Accuracy with Permutation") +
  theme(legend.position="None")

plot_toolbox_ml_accuracy_noscaling

ggsave("plot_toolbox_accuracy_noscaling_ml_01-24-2024.pdf",plot_toolbox_ml_accuracy_noscaling, units="in", width=6.5, height=4.5)


plot_toolbox_ml_md_noscaling <- plot_var_stability(nested_cv, metric_imp="md", final = FALSE, final_fitting=FALSE) +
  coord_cartesian(xlim=c(2,6)) +
  labs(x="Mean Minimal Depth") +
  theme(legend.position="None")

ggsave("plot_toolbox_md_noscaling_ml_01-24-2024.pdf",plot_toolbox_ml_md_noscaling, units="in", width=6.5, height=4.5)


### Create graph of ranked importance by both accuracy and md, with direction marked ----

#### Calculate direction based on uncorrected t-tests or chi squared ----
ttest.pos.affect <- t.test(toolbox_pos_affect_CAT ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p = 0.35
ttest.general.life.sat <- t.test(toolbox_general_life_sat_CAT ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p =0.03
ttest.meaning <- t.test(toolbox_meaning_CAT ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p = 0.04
ttest.emotion.support <- t.test(toolbox_emotion_support ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p < 0.001
ttest.instrument.support <- t.test(toolbox_instrument_support ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p = 0.01
ttest.friendship <- t.test(toolbox_friendship ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p = 0.01
ttest.loneliness <- t.test(toolbox_loneliness ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p = 0.004
ttest.perceive.reject <- t.test(toolbox_perceiv_reject ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p < 0.001
ttest.perceive.hostile <- t.test(toolbox_perceiv_hostile ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p = 0.15
ttest.self.effic <- t.test(toolbox_self_effic_CAT ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p = 0.001
ttest.perceive.stress <- t.test(toolbox_perceiv_stress ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p < 0.001
ttest.fear.affect <- t.test(toolbox_fear_affect_CAT ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p = 0.046
ttest.fear.somatic.arouse <- t.test(toolbox_fear_somatic_arouse ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p = 0.059
ttest.sad <- t.test(toolbox_sad_CAT ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p = 0.01
ttest.anger.affect <- t.test(toolbox_anger_affect_CAT ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p = 0.20
ttest.anger.hostile <- t.test(toolbox_anger_hostile ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p < 0.001
ttest.anger.physical <- t.test(toolbox_anger_physical ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p = 0.008

ttest.age <- t.test(age_in_years ~ suicide_attempt_group, data=emotional_toolbox_bipolar_only , var.equal=TRUE) # p = 0.9998
chisq.sex <- chisq.test(emotional_toolbox_bipolar_only $sex, emotional_toolbox_bipolar_only $suicide_attempt_group) # p = 0.1477

directionality <- data.frame(variable = sort(names(emotional_toolbox_bipolar_only[-18])),
                             direction = c("No Difference", #age
                                           "No Difference", #sex
                                           "No Difference", #anger-affect
                                           "Up in Attempt+ group", #anger-hostility
                                           "Up in Attempt+ group", #anger-physical
                                           "Down in Attempt+ group", #emotional support
                                           "Up in Attempt+ group", #fear-affect
                                           "No Difference", #fear-somatic
                                           "Down in Attempt+ group", #friendship
                                           "Down in Attempt+ group", #gen life satisfaction
                                           "Down in Attempt+ group", #instrumental support
                                           "Up in Attempt+ group", #loneliness
                                           "Down in Attempt+ group", #meaning
                                           "No Difference", #perceived hostility
                                           "Up in Attempt+ group", #perceived rejection
                                           "Up in Attempt+ group", #perceived stress
                                           "No Difference", #positive affect
                                           "Up in Attempt+ group", #sadness
                                           "Down in Attempt+ group" #self-efficacy
                             ))

#### prepare data frames for variable importance of outer cv accuracy and md ----
cv_varImp_accuracy <- cv_varImp(nested_cv, metric_imp="accuracy", final=FALSE)
cv_varImp_md <- cv_varImp(nested_cv, metric_imp="md", final=FALSE)


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
                                      toolbox_emotion_support = "Emotional support",     
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
  full_join(directionality) %>%
  mutate(direction = factor(direction)) %>%
  mutate(md_rank = dense_rank(mean_md),
         accuracy_rank = dense_rank(desc(mean_accuracy)))
  
importance_cv %>%
  arrange(accuracy_rank) 


plot_toolbox_ml_acc_vs_md_noscaling <- ggplot(data=importance_cv, aes(y=mean_md, x=mean_accuracy, 
                                           color=direction)) +
  geom_label_repel(aes(label=accuracy_rank), max.overlaps=40,
                   label.padding=0.1, force=0.00022, fontface="bold") +
  scale_color_manual(values= c("Down in Attempt+ group" = "blue",
                               "Up in Attempt+ group" = "red", 
                               "No Difference" = "grey")) + 
  labs(x="Outer CV Folds' Mean Decrease in Accuracy",
       y="Outer CV Folds' Mean Minimal Depth",
       color="Directionality") +
  coord_cartesian(xlim=c(-0.003, 0.02)) +
  scale_x_continuous(breaks=c(0, 0.005, 0.01, 0.015, 0.02)) +
  theme_prism() +
  theme(legend.position = "None") 


ggsave("plot_toolbox_acc_vs_md_noscaling_ml_01-24-2024.pdf",plot_toolbox_ml_acc_vs_md_noscaling, units="in", width=6.5, height=4.5)
