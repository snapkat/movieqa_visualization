<!DOCTYPE html>
<html>
% include('header.tpl', rand=1)
<body>
	<div class="container">
	<h2>Error</h2>
	<h3>{{message}}</h3>
	<form method="get" action="/">
	<button id="button_button" class="six columns offset-by-three button-primary" type="submit"> Return</button></form>
	</div>
</body>
</html>
