
create_vid(){
	video_pattern=$1
	video_path=$2
	ffmpeg -hide_banner -v quiet -stats -f image2 -framerate 5 -i $video_pattern -c:v libx264 -preset veryfast  -r 25 -crf 32 -pix_fmt yuv420p -movflags +faststart -y $video_path
}

create_vid imgs/%05d.png video.mp4
create_vid imgs/flow_%05d.png video_flow.mp4
