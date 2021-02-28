srcs=(it pt ro fr es)
for src in "${srcs[@]}"; do
  sh main.sh OpenSubtitles $src ar
  #sh main.sh MultiCCAligned  $src ar
  sh main.sh TED2020 $src ar
done


