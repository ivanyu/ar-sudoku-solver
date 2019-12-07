.PHONY: fonts
fonts: fonts/0cef8205-ubuntu-font-family-0.83.zip fonts/open-sans.zip
	unzip -d fonts/ -o fonts/0cef8205-ubuntu-font-family-0.83.zip ubuntu-font-family-0.83/*.ttf
	mv fonts/ubuntu-font-family-0.83/* fonts/
	rm -r fonts/ubuntu-font-family-0.83/

	unzip -d fonts/ -o fonts/open-sans.zip *.ttf

fonts/0cef8205-ubuntu-font-family-0.83.zip:
	wget https://assets.ubuntu.com/v1/0cef8205-ubuntu-font-family-0.83.zip -P fonts

fonts/open-sans.zip:
	wget https://www.opensans.com/download/open-sans.zip -P fonts

.PHONY: model
model: fonts
	python __main__.py
