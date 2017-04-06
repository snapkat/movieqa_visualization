"use strict";

var imdb_key = $('#imdb-key').html()

var get_poster_local = function(){
	$(this).attr("src", '/poster/'+ imdb_key+'.jpg');
};

function get_poster(){
	$.ajax({
        url: 'http://www.omdbapi.com/?i=' + imdb_key,
        success: show_poster
    });
}

var show_poster = function(res){
	console.log(res);
	$("#poster").attr("src", res.Poster);
	$("#movie-title").html(res.Title)
}

function select_word(){
	window.location.href = $( this ).attr('name');
}

function go_home(){
	window.location.href = '/';
}

$(".plot-word").click(select_word);
$("#poster").click(go_home);

get_poster_local()