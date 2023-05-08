select count(*)
from info_type it1,
info_type it2,
title t,
movie_info mi,
cast_info ci,
name n,
person_info pi
WHERE t.id = ci.movie_id
AND ci.person_id = n.id
AND n.id = pi.person_id
AND it2.info ILIKE '%birth%'
AND pi.info_type_id = it2.id
AND pi.info ILIKE '%india%'
AND it1.info ILIKE '%count%'
AND mi.info_type_id = it1.id
AND t.id = mi.movie_id
AND mi.info ILIKE '%usa%';
