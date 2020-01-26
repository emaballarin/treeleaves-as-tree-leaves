## Print this on start so I know it's loaded -- ON TOP.
cat("Loading custom .Rprofile...\n")


# Set the default mirror of the CRAN & default working directory
.First <- function() {
    options(
        repos = c(CRAN = "https://cran.stat.unipd.it/"),
        setwd("~"))
}


# Require devtools if it could be used
if (interactive()) {
    suppressMessages(require(devtools))
}


# Really personal user preferences
options(tab.width = 4)
options(width = 100)
options(editor="code")
options(papersize="a4")
Sys.setenv(R_HISTSIZE='100000')

# Customize R interactive prompt
options(prompt="> ")
options(continue="... ")


# Useful options to simplify life
options(stringsAsFactors=FALSE)     # self-explanatory
utils::rc.settings(ipck=TRUE)       # allow package name autocompletion


# Colorize output if possible
#if(Sys.getenv("TERM") == "xterm-256color")
#    library("colorout")


## Utility functions ##
.env <- new.env()

## Shorthands
.env$s <- base::summary
.env$h <- utils::head

## 10-row-head 'n' 10-row-tail
.env$ht <- function(d) rbind(head(d,10),tail(d,10))

## List all functions in a package.
.env$lsp <-function(package, all.names = FALSE, pattern) {
    package <- deparse(substitute(package))
    ls(
        pos = paste("package", package, sep = ":"),
        all.names = all.names,
        pattern = pattern
    )
}

.env$unrowname <- function(x) {     # Remove empty rownames
    rownames(x) <- NULL
    x
}

.env$unfactor <- function(df){      # Undo calls to factor()
    id <- sapply(df, is.factor)
    df[id] <- lapply(df[id], as.character)
    df
}

## Attach all the variables above
attach(.env)


## Print this on start so I know it's loaded -- ON BOTTOM.
cat(".Rprofile LOADED!\n\n")
