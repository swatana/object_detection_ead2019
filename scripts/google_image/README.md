Paste the following JavaScript to Google Images on console.
```
	var query = document.querySelector('.gLFyf').value;
	// pull down jquery into the JavaScript console
	var script = document.createElement('script');
	script.src = "https://ajax.googleapis.com/ajax/libs/jquery/2.2.0/jquery.min.js";
	document.getElementsByTagName('head')[0].appendChild(script);
```
```
	// grab the URLs
	var urls = $('.rg_di .rg_meta').map(function() { return JSON.parse($(this).text()).ou; });

	// write the URls to file (one per line)
	var textToSave = urls.toArray().join('\n');
	var hiddenElement = document.createElement('a');
	hiddenElement.href = 'data:attachment/text,' + encodeURI(textToSave);
	hiddenElement.target = '_blank';
	hiddenElement.download = query + '.txt';
	hiddenElement.click();
```
Paste twice if text is not downloaded

Run
```
cp ~/Downloads/横断歩道.txt .
apt install curl
./fetch.sh 横断歩道.txt
```
