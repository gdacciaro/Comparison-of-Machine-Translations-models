wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fBe7syWG0kBrd1zsX6u7nJwtCcLj67QZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Boo9w_mLJeqCilBCZNr5VOGGrbfc202H" -O weights.zip && rm -rf /tmp/cookies.txt

unzip models.zip

rm models.zip