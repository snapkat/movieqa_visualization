"use strict";


$('#tfidf').click(function(){
	var theURL = window.location.pathname;
    return  theURL.replace("/url_part_to_change/", "/new_url_part/");
    window.location.href = window.location.pathname.replace(/^.*[\\\/]/, '')
})