select count(*)
from title t, movie_keyword mk, keyword k, info_type it, movie_info mi
where it.id = 3 AND it.id = mi.info_type_id AND mi.movie_id = t.id AND
mk.keyword_id = k.id AND mk.movie_id = t.id
AND k.keyword ILIKE '%love%'
AND mi.info ILIKE '%romance%';
