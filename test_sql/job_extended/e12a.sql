-- sorted list of top actors from india in given genres
select n.name
from title t,
name n,
cast_info ci,
movie_info mi,
info_type it1,
info_type it2,
person_info pi
WHERE t.id = ci.movie_id
AND t.id = mi.movie_id
AND ci.person_id = n.id
AND it1.id = 3
AND it1.id = mi.info_type_id
AND (mi.info ILIKE '%romance%'
  OR mi.info ILIKE '%action%')
AND it2.info ILIKE '%birth%'
AND pi.person_id = n.id
AND pi.info_type_id = it2.id
AND pi.info ILIKE '%usa%'
group by n.name
order by count(*) DESC
