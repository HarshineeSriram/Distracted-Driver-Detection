train_folders = c('c0' , 'c1' , 'c2' , 'c3' , 'c4' , 'c5' , 'c6' , 'c7' , 'c8' , 'c9')
main_path = 'E:/gaip/Project'
library(dplyr)
library(EBImage)

## Human Skin Colors

colors <-read.table(header=TRUE, sep=",", text="
                    r,g,b
                    0.1,0.1,0
                    0.2,0.2,0.1
                    0.3,0.3,0.3
                    0.3,0.3,0.2
                    0.4,0.4,0.3
                    0.4,0.4,0.4
                    0.2,0.1,0.1
                    0.5,0.5,0.4
                    0.3,0.2,0.2
                    0.4,0.5,0.4
                    0.9,1,0.9
                    0.6,0.7,0.7
                    0.5,0.5,0.3
                    0.4,0.7,0.8
                    0.6,0.6,0.5
                    0.4,0.3,0.3
                    0.5,0.5,0.5
                    0.7,0.7,0.6
                    0.5,0.4,0.3
                    1,1,0.9
                    0.8,0.8,0.7
                    0.5,0.4,0.4
                    0.6,0.6,0.4
                    0.4,0.3,0.2
                    0.8,0.9,0.9
                    0.5,0.6,0.5
                    0.6,0.6,0.6
                    0.6,0.7,0.6
                    0.7,0.7,0.7
                    0.7,0.8,0.7
                    0.7,0.7,0.5
                    0.9,0.9,0.8
                    0.8,0.9,0.8
                    0.7,0.6,0.5
                    0.8,0.7,0.6
                    0.6,0.5,0.4
                    0.9,0.9,0.7
                    0.8,0.8,0.6
                    0.8,0.8,0.8
                    0.9,0.8,0.7
                    0.8,0.7,0.7
                    0.9,0.9,0.9
                    0.9,0.8,0.8
                    0.8,0.7,0.5
                    1,0.9,0.9")

filterthiscolors <- function(x, colors){
  dt <- as.data.frame(round(matrix(x,nrow=prod(dim(x)[1:2]),ncol=3),1))
  colnames(dt) <- c("r","g","b")
  colors$yes <- TRUE
  d1 <- left_join(dt,colors,by=c("r","g","b"))
  d1$yes[is.na(d1$yes)] <- FALSE
  c(d1$yes,d1$yes,d1$yes)
}
  

images_path = 'E:/gaip/Project/images'

for(i in train_folders[1:10]){
  
  train_path = file.path('E:/gaip/Project/images/train', i)
  imgs = list.files(train_path)

  
  for(img_file in imgs){
    file <- paste0(file.path(train_path, img_file))
    x <- readImage(file)
    filter <- filterthiscolors(x,colors)
    table(filter)
    x1 <- x
    x1[!filter] <- 0 
    x2 <- medianFilter(x1,4)
    x3 <- x
    x3[as.array(x2) < 0.25 | as.array(x2) > 0.8] <- 0
    
    pp_save_path = file.path('E:/gaip/Project/filtered images', i)
    
    mypath <- file.path(pp_save_path, img_file)
    jpeg(file=mypath)
    plot(x3)
    dev.off()
  }
}


