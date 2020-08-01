library(ggplot2)
library(readxl)
library(plyr)

setwd('~/Projects/arrhythmia-detection')

data_dir <- 'data/raw/'
plot_dir <- 'plots/'

diagnostics <- read_xlsx(file.path(data_dir, 'Diagnostics.xlsx'))
diagnostics$Rhythm <- as.factor(diagnostics$Rhythm)
diagnostics$Beat <- as.factor(diagnostics$Beat)
diagnostics$PatientAge <- as.numeric(diagnostics$PatientAge)
diagnostics$Gender <- as.factor(diagnostics$Gender)

# Check label distribution

name <- 'hist_rhythm.pdf'
width <- 4
height <- 2

plot <- ggplot(diagnostics) +
  geom_bar(aes(Rhythm)) +
  ylab('Count') +
  theme_minimal() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

plot

pdf(file=file.path(plot_dir, name),width=width, height=height)
plot
dev.off()


# Check gender distribution

name <- 'hist_gender.pdf'
width <- 4
height <- 1.5

plot <- ggplot(diagnostics) +
  geom_bar(aes(Gender)) +
  ylab('Count') +
  theme_minimal() +
  coord_flip()

plot

pdf(file=file.path(plot_dir, name),width=width, height=height)
plot
dev.off()

# Check age distribution

name <- 'hist_age.pdf'
width <- 4
height <- 2

plot <- ggplot(diagnostics) +
  geom_histogram(aes(PatientAge), bins = 20) +
  geom_density(aes(PatientAge, y=5 * ..count..), colour='red') +
  ylab('Count') +
  theme_minimal() +
  theme(axis.title.x = element_blank())
plot
    
pdf(file=file.path(plot_dir, name),width=width, height=height)
plot
dev.off()

print(mean(diagnostics$PatientAge))
print(sd(diagnostics$PatientAge))


# Map Tachycardia label and remove uncommon ones
previous <- nrow(diagnostics)

diagnostics$Rhythm <- mapvalues(diagnostics$Rhythm, 
                               from=c('SVT','AT','AVNRT'), 
                               to=c('ST','ST','ST' ))

diagnostics = diagnostics[!diagnostics$Rhythm %in% c('AF', 'AVRT', 'SA', 'SAAWR'), ]

print('Dropped rows:')
print(previous - nrow(diagnostics))

name <- 'hist_rhythm_clean.pdf'
width <- 4
height <- 2

plot <- ggplot(diagnostics) +
  geom_bar(aes(Rhythm)) +
  ylab('Count') +
  theme_minimal() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

plot

pdf(file=file.path(plot_dir, name),width=width, height=height)
plot
dev.off()


# Boxplot of age to rhythm

name <- 'box_age_rhythm.pdf'
width <- 4.2
height <- 3

plot <- ggplot(diagnostics) +
  geom_boxplot(aes(PatientAge, Rhythm, fill=Gender)) +
  xlab('Patient Age') +
  theme_minimal() +
  theme(legend.position="bottom",
        legend.title = element_blank())

plot

pdf(file=file.path(plot_dir, name),width=width, height=height)
plot
dev.off()
