SELECT MIN(chn.name) AS uncredited_voiced_character, MIN(t.title) AS russian_movie 
FROM char_name chn, cast_info ci, movie_companies mc, title t
WHERE ci.note  like '%(voice)%' AND ci.note like '%(uncredited)%' AND t.production_year > 2003 
AND t.id = mc.movie_id AND t.id = ci.movie_id AND ci.movie_id = mc.movie_id AND chn.id = ci.person_role_id ;
