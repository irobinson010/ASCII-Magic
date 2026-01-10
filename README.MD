This project is used to take images turn them into ASCII and then colorize them.

Use https://www.asciiart.eu/ to help turn your image into ASCII I hope to imulate this in the project later to have it in one place.

Turn image(png please) into ASCII https://www.asciiart.eu/image-to-ascii
Make Text into ASCII pretty print https://www.asciiart.eu/text-to-ascii-art

Note that the images are best to turn into a cartoon version or an anime version as it simplifies the image shape and color to make it easier to show in ASCII. You can use GIMP filters to change them or just put it into an Ai system and tell it to give you a Charachter version or Anime version of the image.

Now that you have turned your image into ASCII and saved it to a file. You can make your ASCII text and put it at the top of the file if you would like(take note of how many rows it takes up)

Now the fun part take the png and the ASCII and color it with colorize_ascii.py
do this simply by calling the colorize_ascii.py(make sure it is executable chmod +x filehere) then the first parameter is the png image followed by the ASCII text file then the output file whatever you want .ans. then you can add the other options like --keep-top 5 this leaves the top 5 rows alone when scaling the image to your prefered size leaving your text ASCII Alone. The other options can be seen with just running ./colorize_ascii.py Below is an example of running it.

./colorize_ascii.py ~/Downloads/input-image.png ~/Downloads/input-ASCII.txt ./output-colored.ans --max-rows 30  --keep-top 5 --color-top
