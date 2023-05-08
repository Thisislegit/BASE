--- worst rated movies of american actors
select min(n.name), min(t.title)
from info_type it1,
info_type it2,
movie_info_idx mii,
title t,
cast_info ci,
name n,
person_info pi
WHERE it1.info ILIKE 'rating'
AND it1.id = mii.info_type_id
AND t.id = mii.movie_id
AND t.id = ci.movie_id
AND ci.person_id = n.id
AND n.id = pi.person_id
AND pi.info_type_id = it2.id
AND it2.info ILIKE '%birth%'
AND pi.info ILIKE '%USA%'
AND (mii.info ILIKE '0%' OR mii.info ILIKE '1%'
  OR mii.info ILIKE '2%');

-- AND mii.info::int > 1000;
-- order by mii.info::int desc LIMIT 100;
