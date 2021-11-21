import React from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom';
import about from './about';
import Post from './post';
import './index.css';

ReactDOM.render(
    <Router>
      <Switch>
        <Route exact path="/" component={about} />

        <Route path="/:id" component={Post} />
      </Switch>
    </Router>,

  document.getElementById('root'));