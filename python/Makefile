.PHONY: fonts
#fonts: fonts/0cef8205-ubuntu-font-family-0.83.zip fonts/open-sans.zip
fonts: fonts/FiraCode_2.zip fonts/roboto-android.zip fonts/0cef8205-ubuntu-font-family-0.83.zip fonts/imprimerie.zip fonts/belmist.zip
	unzip -d fonts/ -o fonts/FiraCode_2.zip ttf/*.ttf
	mv fonts/ttf/* fonts/
	rm -r fonts/ttf

	unzip -d fonts/ -o fonts/belmist.zip *.ttf

	unzip -d fonts/ -o fonts/imprimerie.zip *.ttf

	unzip -d fonts/ -o fonts/roboto-android.zip *.ttf

	unzip -d fonts/ -o fonts/0cef8205-ubuntu-font-family-0.83.zip ubuntu-font-family-0.83/*.ttf
	mv fonts/ubuntu-font-family-0.83/* fonts/
	rm -r fonts/ubuntu-font-family-0.83/

#	unzip -d fonts/ -o fonts/open-sans.zip *.ttf

fonts/FiraCode_2.zip:
	wget https://github.com/tonsky/FiraCode/releases/download/2/FiraCode_2.zip -P fonts

fonts/roboto-android.zip:
	wget https://github.com/google/roboto/releases/download/v2.138/roboto-android.zip -P fonts

fonts/open-sans.zip:
	wget https://www.opensans.com/download/open-sans.zip -P fonts

fonts/0cef8205-ubuntu-font-family-0.83.zip:
	wget https://assets.ubuntu.com/v1/0cef8205-ubuntu-font-family-0.83.zip -P fonts

fonts/imprimerie.zip:
	wget https://dl.dafont.com/dl/?f=imprimerie -O fonts/imprimerie.zip

fonts/belmist.zip:
	wget https://dl.dafont.com/dl/?f=belmist -O fonts/belmist.zip
